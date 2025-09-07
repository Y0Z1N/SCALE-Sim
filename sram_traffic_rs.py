import math 
from tqdm import tqdm

# V1. Unlimited RF size
# - Given one ifmap, computation flows like:
# / For one filter to num_filt:
# /     <- Ofmap Folding may happen ->
# /     For one channel to num_channels:
# /         <- Filt Folding may happen ->
# /         compute_2d_conv()
# /         <--------------------------->
# /     acc_channel_psums()
# /     <----------------------------->

# In Row Stationary, there's single SRAM (scratchpad mem)
# In each 'compute_2d_convolution', there may be several folds.
# Then 'ifmap reuse' is followed, so SRAM always contains one ifmap data.
# But under unlimited RF size, it can happen 'channel accumulation' in PE.

# Before one 'compute_2d_conv()', SRAM must be prepared one filter's one channel data at least.
# In summary, in each fold, SRAM holds
# 1) One ifmap data (pinned for whole 3D conv)
# 2) one filter

def get_1d_idx_filt(curr_filt,
                    num_chan, curr_chan,
                    filt_h, curr_row,
                    filt_w, curr_col):
    return ((((curr_filt * num_chan + curr_chan)
                 * filt_h) + curr_row) * filt_w) + curr_col

def get_1d_idx_ifmap(curr_channel, ifmap_h, curr_row, ifmap_w, curr_col):
    return ((curr_channel * ifmap_h + curr_row) * ifmap_w) + curr_col



def sram_traffic(
        arr_h=4,                # systolic array num rows
        arr_w=4,                # systolic array num cols
        ifmap_h=7, ifmap_w=7,
        filt_h=3, filt_w=3,
        num_channels=3,
        strides=1, num_filt=8,
        ofmap_base=2000000, filt_base=1000000, ifmap_base=0,
        sram_read_trace_file="sram_read.csv",
        sram_write_trace_file="sram_write.csv",
        rf_size = 1000000 # update later, currently unlimited size
    ):

    # Parameters
    ofmap_h = math.floor((ifmap_h - filt_h + strides) / strides)
    ofmap_w = math.floor((ifmap_w - filt_w + strides) / strides)
    px_per_filt = (filt_h * filt_w * num_channels)

    # 1D-Conv Primitives (prim)
    # 'each primitive operates on one row of filter weights and
    #  one row of ifmap pixels, and generates one row of psums.'

    # Fold Parameters
    # Ofmap row fold is triggered when ofmap_h > arr_w
    # Filt row fold is triggered when filt_h > arr_h
    num_ofmap_fold = int(math.ceil(ofmap_h / arr_w))
    num_filt_fold = int(math.ceil(filt_h / arr_h))
    num_chan_fold = calc_channel_fold(rf_size, ifmap_w, filt_w, num_channels)
    cnt_chan_per_pold = int(num_channels / num_chan_fold)

    # Filter indexing : (num_filter * num_channel) * ...
    # Ifmap indexing : (num_channel) * ...
    
    cycle = 0
    prev_cycle = 0
    compute_cycles = 0

    for curr_filt in range(num_filt):
        1
        for ofmap_fold in range(num_ofmap_fold):
            1
            ofmap_start_idx = (ofmap_fold) * arr_w
            ofmap_offset = arr_w
            
            # Last fold may be lesser than arr_w
            if ofmap_fold == num_ofmap_fold - 1:
                ofmap_offset = ofmap_h % arr_w

            for filt_fold in range(num_filt_fold):
                # Channel fold -> Each PE computes one psum row by interleaving several channels
    
                for chan_fold in range(num_chan_fold):
                    filt_start_idx = get_1d_idx_filt(curr_filt, num_channels, chan_fold,
                                                        filt_h, filt_fold * arr_h, filt_w, 0)
                    filt_offset = arr_h

                    ifmap_start_idx = get_1d_idx_ifmap(chan_fold, ifmap_h,
                                                        ofmap_start_idx + filt_fold * arr_h, ifmap_w, 0)
                    ifmap_offset = arr_h + arr_w - 1

                    # Last fold may be lesser than arr_h
                    if filt_fold == num_filt_fold - 1:
                        filt_offset = filt_h % arr_h
                        ifmap_offset = filt_offset + arr_w - 1
                    
                    # Fill the required data into each PE
                    cycle = gen_read_trace(cycle, arr_h, arr_w, ifmap_h, ifmap_w, filt_h, filt_w, 
                                           cnt_chan_per_pold, strides, ifmap_start_idx, ifmap_offset, 
                                           filt_start_idx, filt_offset, ifmap_base, filt_base, sram_read_trace_file)
                    
                    # Compute the cycles that each PE needs for one primitive
                    cycle += compute_prim_cycle(ofmap_w, cnt_chan_per_pold, strides)

                    # Now uppermost PE accumulates the one column of its array
                    cycle += (arr_h - 1)

                    # Write the ofmap psum on SRAM
                    cycle = gen_write_trace(cycle, arr_h, arr_w, ifmap_h, ifmap_w, filt_h, filt_w,
                                            cnt_chan_per_pold, strides, ofmap_start_idx, ofmap_offset, 
                                            ofmap_base, sram_write_trace_file)

                    # calc util of this primitive fold
                    delta_cycle = cycle - prev_cycle
                    util_this_fold = (filt_offset * ofmap_offset) / (arr_h * arr_w)

                    compute_cycles += delta_cycle
                    util += util_this_fold * delta_cycle
                    prev_cycle = cycle

            1
            # Now one row of one ofmap should be computed (accumulation)


    final = str(cycle)
    final_util = (util / compute_cycles) * 100

    return (final, final_util)


def gen_read_trace(
        cycle = 0,
        arr_h = 4, arr_w = 4,
        ifmap_h = 7, ifmap_w = 7,
        filt_h = 3, filt_w = 3,
        num_channels = 3, strides = 1,
        ifmap_start_idx = 0, ifmap_offset = 0,
        filt_start_idx = 0, filt_offset = 0,
        ifmap_base = 0, filt_base = 1000000,
        sram_read_trace_file = "sram_read.csv"
):
    outfile = open(sram_read_trace_file, 'a')

    # We assume that row stationary 'broadcasts' each data to entire PE

    entry = str(cycle) + ", "

    # Padding 추가할 것 (마지막 fold가 arr 을 다 채우지 못 하는 경우)

    # Ifmap read address
    for i in range(ifmap_offset):
        entry += str(ifmap_base + ifmap_start_idx + i) + ", "
    
    # Filt read address
    for f in range(filt_offset):
        entry += str(filt_base + filt_start_idx + f) + ", "
    
    entry += "\n"

    outfile.write(entry)
    outfile.close()
    return cycle+1

    
def gen_write_trace(
        cycle = 0,
        arr_h = 4, arr_w = 4,
        ifmap_h = 7, ifmap_w = 7,
        filt_h = 3, filt_w = 3,
        num_channels = 3, strides = 1,
        ofmap_start_idx = 0, ofmap_offset = 0,
        ofmap_base = 2000000,
        sram_write_trace_file = "sram_write.csv"
):
    outfile = open(sram_write_trace_file, 'a')
    
    entry = str(cycle) + ", "

    for o in range(ofmap_offset):
        entry += str(ofmap_base + ofmap_start_idx + o) + ", "

    entry += "\n"
    
    outfile.close()
    return cycle


def compute_prim_cycle(
        ofmap_w = 5,
        cnt_chan_per_fold = 1,
        strides = 1
):
    # Further implementation may needs
    # if there happens 'ifmap stream folds'
    # Currently, we assume whole ifmap row value is saved in RF
    prim_cycle = ofmap_w * cnt_chan_per_fold
    return prim_cycle


def calc_channel_fold(
        rf_size = 10000000,
        ifmap_w = 7, filt_w = 3,
        num_channels = 3
):
    # Update later
    return 1