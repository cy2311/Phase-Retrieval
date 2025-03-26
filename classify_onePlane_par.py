import numpy as np
from joblib import Parallel, delayed

def classify_onePlane_par(subregion_ch1, ref_plane1, empupil):
    """
    Classify single molecules to reference Z-position images and average these Z-position images in a certain group.

    Parameters:
    subregion_ch1 (numpy.ndarray): Single molecule images (3D array: height x width x num_images).
    ref_plane1 (numpy.ndarray): Reference Z-position images (3D array: height x width x num_ref).
    empupil (dict): Dictionary containing parameters:
        - 'imsz': Size of the images.
        - 'min_similarity': Minimum similarity threshold.
        - 'bin_lowerBound': Minimum number of spots required in a group.

    Returns:
    ims_Zcal_ave_plane1 (numpy.ndarray): Averaged images for each Z-position (3D array: height x width x num_ref).
    index_record_Zplanes (numpy.ndarray): Indices of Z-positions with valid averaged images (1D array: num_ref).
    """
    print('Calculate the similarity between reference images and single molecules')
    img_plane1 = subregion_ch1.astype(np.float32)  # Normalization image
    num_img = img_plane1.shape[2]
    num_ref = ref_plane1.shape[2]
    imsz = empupil['imsz']

    # Initialize arrays
    similarity_in_Plane1 = np.zeros((num_ref, num_img))  # Similarity
    index_similarity = np.zeros((num_ref, num_img))  # Index
    shift_row_plane1 = np.zeros((num_ref, num_img))  # X shift in plane1
    shift_col_plane1 = np.zeros((num_ref, num_img))  # Y shift in plane1

    # Parallel computation of similarity and shifts
    def process_image(ii):
        similarity_row = np.zeros(num_ref)
        shift_row = np.zeros(num_ref)
        shift_col = np.zeros(num_ref)
        for jj in range(num_ref):
            shift1, shift2, tmpval1 = registration_in_each_channel(ref_plane1[:, :, jj], img_plane1[:, :, ii])
            if abs(shift1) > 6 or abs(shift2) > 6:  # This value can be changed, now fixed
                continue
            shift_row[jj] = shift1
            shift_col[jj] = shift2
            if tmpval1 < empupil['min_similarity']:
                continue
            similarity_row[jj] = tmpval1
        return similarity_row, shift_row, shift_col

    results = Parallel(n_jobs=-1)(delayed(process_image)(ii) for ii in range(num_img))
    for ii, (similarity_row, shift_row, shift_col) in enumerate(results):
        similarity_in_Plane1[:, ii] = similarity_row
        shift_row_plane1[:, ii] = shift_row
        shift_col_plane1[:, ii] = shift_col

    # Determine which reference image each single molecule belongs to
    for ii in range(num_img):
        sort_similarity = np.sort(similarity_in_Plane1[:, ii])[::-1]
        index_sort = np.argsort(similarity_in_Plane1[:, ii])[::-1]
        if sort_similarity[0] == 0:
            continue
        index_similarity[index_sort[0], ii] = 1
        for jj in range(1, num_ref):
            if sort_similarity[jj] >= sort_similarity[0] - 0.0 and abs(index_sort[jj] - index_sort[0]) == 1:  # Now fixed
                index_similarity[index_sort[jj], ii] = 1
            else:
                break

    # Update average images
    print('Update average images')
    ims_Zcal_ave_plane1 = np.zeros((imsz, imsz, num_ref))
    index_record_Zplanes = np.zeros(num_ref)
    for ii in range(num_ref):
        index_selection = np.where(index_similarity[ii, :] == 1)[0]
        sz_index = index_selection.size
        if sz_index > empupil['bin_lowerBound']:  # Each group must have enough spots
            ims_plane1_shift = np.zeros((imsz, imsz, sz_index))
            for jj in range(sz_index):
                ims_plane1_shift[:, :, jj] = FourierShift2D(
                    similarity_in_Plane1[ii, index_selection[jj]] * img_plane1[:, :, index_selection[jj]],
                    [shift_row_plane1[ii, index_selection[jj]], shift_col_plane1[ii, index_selection[jj]]]
                )
            # Average the images
            index_record_Zplanes[ii] = 1
            ims_Zcal_ave_plane1[:, :, ii] = np.mean(ims_plane1_shift, axis=2)

    # Remove too far away reassembled PSF
    tmp_record_keep = np.where(index_record_Zplanes == 1)[0]
    size_tmp = tmp_record_keep.size
    for ii in range(size_tmp // 2, 0, -1):
        if tmp_record_keep[ii] > tmp_record_keep[ii - 1] + 2:
            index_record_Zplanes[tmp_record_keep[ii - 1]] = 0
            tmp_record_keep[ii - 1] = tmp_record_keep[ii]
    for ii in range(size_tmp // 2, size_tmp - 1):
        if tmp_record_keep[ii] < tmp_record_keep[ii + 1] - 2:
            index_record_Zplanes[tmp_record_keep[ii + 1]] = 0
            tmp_record_keep[ii + 1] = tmp_record_keep[ii]

    return ims_Zcal_ave_plane1, index_record_Zplanes
