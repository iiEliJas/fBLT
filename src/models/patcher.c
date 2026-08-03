#include "blt/models/patcher.h"
#include "blt/core/backend.h"


/*
    Starts a new patch when the entropy exceeds the threshold, and closes the current patch when the entropy falls below the threshold
    It also tracks the peak entropy value for each patch
*/
size_t blt_segment_patches(const blt_tensor* entropy, blt_patch_info* patches_out, size_t max_patches, const blt_entropy_config* config){
    if(entropy->ndim != 1){
        BLT_FATAL("Entropy tensor must be 1D");
    }
    if(entropy->numel == 0){
        BLT_FATAL("Entropy tensor is empty");
    }

    size_t seq_len = entropy->numel;
    size_t patch_count = 0;

    size_t current_patch_start = 0;
    float current_patch_peak_entropy = ((float*)entropy->data)[0];
    
    for(size_t i = 1; i < seq_len; ++i){
        float current_entropy = ((float*)entropy->data)[i];
        if(current_entropy >= config->threshold){
            // Close current patch
            if(patch_count < max_patches){
                patches_out[patch_count].start_idx = current_patch_start;
                patches_out[patch_count].length = i - current_patch_start;
                patches_out[patch_count].peak_entropy = current_patch_peak_entropy;
                patch_count++;
            } else {
                BLT_WARN("Reached maximum number of patches (%zu)", max_patches);
                break;
            }
            // Start new patch
            current_patch_start = i;
            current_patch_peak_entropy = current_entropy;
        } else {
            // Update peak entropy for current patch
            if(current_entropy > current_patch_peak_entropy){
                current_patch_peak_entropy = current_entropy;
            }
        }
    }

    // Close the final patch
    if(patch_count < max_patches){
        patches_out[patch_count].start_idx = current_patch_start;
        patches_out[patch_count].length = seq_len - current_patch_start;
        patches_out[patch_count].peak_entropy = current_patch_peak_entropy;
        patch_count++;
    } else {
        BLT_WARN("Reached maximum number of patches (%zu) while closing final patch", max_patches);
    }

    return patch_count;
}