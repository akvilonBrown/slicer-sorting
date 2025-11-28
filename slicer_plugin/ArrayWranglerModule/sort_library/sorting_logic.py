import numpy as np
import slicer

# pylint: skip-file

# check if scikit-learn is installed, if not - install it
try:
    from sklearn.cluster import KMeans
except ModuleNotFoundError:
    if slicer.util.confirmOkCancelDisplay(
        "This module requires 'scikit-learn' Python package. "
        "Click OK to install it now."
    ):
        slicer.util.pip_install("scikit-learn")
        from sklearn.cluster import KMeans

# dictinary defining sorting order (axes) in the numpy array
sorting_order_classic = {
    "height": 2,  # the third axis in a 3D numpy array z
    "height_direction": 1,  # ascending
    "rows": 0,  # the first axis in a 3D numpy array y
    "rows_direction": 1,  # ascending
    "columns": 1,  # the second axis in a 3D numpy array x
    "columns_direction": 1,  # ascending
    "rotation_z": (1, 0),  # rotation around z axis
    "rotation_x": (2, 0),  # rotation around x axis
    "rotation_y": (1, 2),  # rotation around y axis
}

# dictinary defining sorting order (axes) in the numpy array
# with sorting order compatible to RAS view
sorting_order_ras = {
    "height": 0,  # the first axis in a RAS adapted numpy array
    "height_direction": -1,  # descending
    "rows": 1,  # the second axis in a RAS adapted numpy array
    "rows_direction": -1,  # descending
    "columns": 2,  # the third axis in a RAS adapted numpy array
    "columns_direction": 1,  # ascending
    "rotation_z": (1, 2),  # rotation around z axis
    "rotation_x": (0, 1),  # rotation around x axis
    "rotation_y": (0, 2),  # rotation around y axis
}


def cluster_zcoord(cpoints, num_zclusters=4, sorder=sorting_order_classic, debug=True):
    height_axis = sorder["height"]
    zdata = cpoints[:, height_axis].reshape(
        -1, 1
    )  # taking only z coordinates in (n, 1) shape
    # zdata = cpoints[:, 0].reshape(-1, 1) # taking only z coordinates in (n, 1) shape
    clustered_obj = KMeans(n_clusters=num_zclusters, random_state=0).fit(zdata)
    z_centers = np.round(clustered_obj.cluster_centers_.flatten(), 0).astype(np.int32)

    orig_labels = clustered_obj.labels_
    # print(orig_labels)
    if debug:
        print("Distribution of labels: label, count (unsorted)")
        for i in np.unique(orig_labels):
            print(i, np.count_nonzero(orig_labels == i))

    # sorting zcluster centers from top to bottom and
    sorted_indx = np.argsort(z_centers)
    if sorder["height_direction"] < 0:  # if reversed
        sorted_indx = sorted_indx[::-1]
    lb_mapping = {
        old_label: new_label for new_label, old_label in enumerate(sorted_indx)
    }
    labels_zsorted = np.array([lb_mapping[label] for label in orig_labels])

    if debug:
        print("Distribution of labels: label, count (sorted)")
        for i in np.unique(labels_zsorted):
            print(i, np.count_nonzero(labels_zsorted == i))

    return labels_zsorted


"""
Function to filter incoming lists by some given value
level - value to filter on
center_points - an array of the shape (n, 3) of points
mapped_labels  - an array of the shape (n,)
with mapping of the points to some (level) value
center_original_labels - an array of the shape (n,)
with mapping of original labels to the points

at the output there will be two arrays with points (f,3) and original labels (f,)
filtered from the whole initial list

"""


def filter_data(level, center_points, mapped_labels, center_original_labels):
    filter_slice = mapped_labels == level
    center_points_filtered = center_points[np.argwhere(filter_slice)[:, 0]]
    center_labels_filtered = center_original_labels[np.argwhere(filter_slice)[:, 0]]
    return center_points_filtered, center_labels_filtered


def level_sort(
    cpoints_level,
    enum_labels_level,
    row_clusters=3,
    mapping_base=100,
    sorder=sorting_order_classic,
    debug=True,
):
    # sorting points by rows coordinate
    # axis/coordinate - by rows - y(0) in the classic array
    # differs at RAS compatiple scheme
    rows_axis = sorder["rows"]
    ydata = cpoints_level[:, rows_axis].reshape(
        -1, 1
    )  # taking only y coordinates in (n, 1) shape
    clustered_obj = KMeans(n_clusters=row_clusters, random_state=0).fit(ydata)
    y_centers = np.round(clustered_obj.cluster_centers_.flatten(), 0).astype(np.int32)

    orig_labels = clustered_obj.labels_
    if debug:
        print(f"{y_centers = }")
        print(f"{orig_labels = }")
        print("Distribution of labels: label, count (unsorted)")
        for i in np.unique(orig_labels):
            print(i, np.count_nonzero(orig_labels == i))

    # sorting ycluster centers from top to bottom and
    sorted_indx = np.argsort(y_centers)
    if sorder["rows_direction"] < 0:  # if order is reversed
        sorted_indx = sorted_indx[::-1]
    lb_mapping = {
        old_label: new_label for new_label, old_label in enumerate(sorted_indx)
    }
    labels_ysorted = np.array([lb_mapping[label] for label in orig_labels])

    labels_ysorted_unique = np.unique(labels_ysorted)
    count_on_rows = [
        np.count_nonzero(labels_ysorted == i) for i in np.unique(labels_ysorted_unique)
    ]

    if debug:

        print("Distribution of labels: label, count (sorted)")
        print(f"{labels_ysorted = }")
        print(f"{labels_ysorted_unique = }")
        print(f"{count_on_rows = }")
        # for i in np.unique(labels_ysorted):
        #    print(i, np.count_nonzero(labels_ysorted == i))

    remap_counter = mapping_base
    remap_dic = {}

    # starting to sort ascending on every row
    # actually i and row_label are supposed to be the same
    for i, row_label in enumerate(labels_ysorted_unique):
        centroids_onrow, enum_labels_onrow = filter_data(
            level=row_label,
            center_points=cpoints_level,
            mapped_labels=labels_ysorted,
            center_original_labels=enum_labels_level,
        )
        # x coordinates (axis 1) - along the row from left to right (columns)
        # in the classical sorting scheme
        # differs in RAS compatible scheme
        column_axis = sorder["columns"]
        xdata = centroids_onrow[:, column_axis]
        sorted_ixx = np.argsort(xdata)
        if sorder["columns_direction"] < 0:  # if reversed
            sorted_ixx = sorted_ixx[::-1]
        enum_labels_onrow_sorted = enum_labels_onrow[sorted_ixx]

        for i, old_label in enumerate(enum_labels_onrow_sorted):
            remap_counter += 1
            remap_dic[old_label] = remap_counter
        # lb_mapping = {old_label: remap_counter+i for i, old_label
        # in enumerate(enum_labels_onrow_sorted)}

        """
        print(f"Distribution of labels in row {row_label}:")
        print(f"{centroids_onrow = }")
        print(f"{enum_labels_onrow = }")

        print(f"{xdata = }")
        print(f"{sorted_ixx = }")
        print(f"{enum_labels_onrow_sorted = }")
        """

    return remap_dic


def full_remap(
    level_bases,
    center_points,
    levelwise_labels,
    init_enum,
    rows_onlevel=3,
    sorting_scheme=sorting_order_classic,
    debug=True,
):
    """
    Function to remap all labels in the whole list of points
    with the mapping dictionary obtained from level_sort function
    level_bases - list of level bases (100, 200, 300, ...)
    center_points - an array of the shape (n, 3) of label centers
    levelwise_labels - an array of the shape (n,) with mapping of the points
    to level value
    init_enum - an array of the shape (n,) with original labels (mapped to the points)
    rows_onlevel - number of clusters (number of rows) on each level (default = 3)
    sorting_sheme - dictionary with sorting preferences
    debug - if True, prints additional information for debugging purposes
    Returns a dictionary with remapped labels
    """

    full_map = {}

    for i, mbase in enumerate(level_bases):
        pts_level, labels_lvl = filter_data(
            level=i,
            center_points=center_points,
            mapped_labels=levelwise_labels,
            center_original_labels=init_enum,
        )
        lmapper = level_sort(
            cpoints_level=pts_level,
            enum_labels_level=labels_lvl,
            row_clusters=rows_onlevel,
            mapping_base=mbase,
            sorder=sorting_scheme,
            debug=debug,
        )
        full_map = full_map | lmapper
    return full_map


def perform_remap(remapping_dict, enum_img):
    """
    Function to remap the labels in the label image using the provided remapping
    dictionary.
    remapping_dict - a dictionary with old labels as keys and new labels as values
    enum_img - the label image to be remapped (numpy array) - passed by value(copy)
    Returns the remapped label image.
    """
    for key in remapping_dict.keys():
        enum_img = np.where(enum_img == key, remapping_dict[key], enum_img)
    return enum_img


def make_consequtive_labels(enum_img, sparse_labels):
    """
    Function to make the labels in the label image consecutive.
    enum_img - the label image to be remapped (numpy array) - passed by value(copy)
    sparse_labels - a list of labels to be remapped into consecutive labels
    Returns the remapped label image.
    """
    for i, labelValue in enumerate(sparse_labels):
        enum_img = np.where(enum_img == labelValue, i + 1, enum_img)
    return enum_img


def expand_object_dims(obj_list, span, ymax0, xmax1, zmax2):
    """
    Function to expand the dimensions of marked blobs in a 3D array.    "
    "obj_list - a list of tuples with slices for each dimension (y, x, z)
    "span - the number of voxels to expand in each direction (default=5)            "
    "Returns a list of expanded slices for each object in the input list."
    """
    ob_expanded = []
    # we expand the dimensions of marked blobs
    for i, ob in enumerate(obj_list):
        if ob is not None:
            yslice0 = slice(
                max(0, ob[0].start - span), min(ymax0, ob[0].stop + span), None
            )
            xslice1 = slice(
                max(0, ob[1].start - span), min(xmax1, ob[1].stop + span), None
            )
            zslice2 = slice(
                max(0, ob[2].start - span), min(zmax2, ob[2].stop + span), None
            )
            ob_expanded.append((yslice0, xslice1, zslice2))

    return ob_expanded


def pad_volume(img, maxdims):
    x_center = img.shape[0] // 2
    xpad1 = maxdims[0] // 2 - x_center
    xpad2 = maxdims[0] - (xpad1 + img.shape[0])

    y_center = img.shape[1] // 2
    ypad1 = maxdims[1] // 2 - y_center
    ypad2 = maxdims[1] - (ypad1 + img.shape[1])

    z_center = img.shape[2] // 2
    zpad1 = maxdims[2] // 2 - z_center
    zpad2 = maxdims[2] - (zpad1 + img.shape[2])

    img = np.pad(
        img,
        ((xpad1, xpad2), (ypad1, ypad2), (zpad1, zpad2)),
        "constant",
        constant_values=0,
    )

    return img


"""
# Reportiing to dataframe
slice1_start, slice1_stop, slice2_start = [],[],[]
slice2_stop, slice3_start, slice3_stop = [],[],[]
for i, ob in zip(labels_index, ob_expanded):
    slice1_start.append(ob[0].start)
    slice1_stop.append(ob[0].stop)
    slice2_start.append(ob[1].start)
    slice2_stop.append(ob[1].stop)
    slice3_start.append(ob[2].start)
    slice3_stop.append(ob[2].stop)
slice_data = {
    'slice1_start':slice1_start,
    'slice1_stop':slice1_stop,
    'slice2_start':slice2_start,
    'slice2_stop':slice2_stop,
    'slice3_start':slice3_start,
    'slice3_stop':slice3_stop,
}
#for_index = list(np.arange(1, len(slice1_start)+1))
df = pd.DataFrame(slice_data, index = labels_index)
"""


def break_cubicles(slices_list, image):
    """
    Function to carve objects from a big volume
    based on the list of slices
    """
    obcubes = []
    shapes = []
    for obex in slices_list:
        obcube = image[obex]
        shapes.append(obcube.shape)
        # print(f"{obcube.shape = }")
        obcubes.append(obcube)
    return obcubes, shapes


def draw_axes(vol_image, offset=3, line_width=1):
    """
    Function to draw axes in a 3D volume image
    vol_image - a 3D numpy array representing the volume image (helper object)
    offset - the offset from the edges where the axes will be drawn (default=3)
    line_width - the width of the axes lines (default=1)
    """
    max0, max1, max2 = vol_image.shape
    # draw_values = [1,2,3]
    # drawing axes, 0 axis is the longest, the 1 axis is half, the 2 is third")
    # vol_image[:, offset:offset+line_width, offset:offset+line_width] = 1
    # vol_image[offset:offset+line_width, :max1//2, offset:offset+line_width] = 2
    # vol_image[offset:offset+line_width, offset:offset+line_width, :max2//3] = 3

    # drawing axes, height z axis2 is the longest, the rows y axis0 is half,
    # the x axis1 is the shortest"
    vol_image[offset : offset + line_width, offset : offset + line_width, :] = 3
    vol_image[
        : max0 // 2, offset : offset + line_width, offset : offset + line_width
    ] = 1
    vol_image[
        offset : offset + line_width, : max1 // 3, offset : offset + line_width
    ] = 2
    return vol_image


def draw_axes_ras(vol_image, offset=3, line_width=1):
    """
    Function to draw axes in a 3D volume image adapted to RAS orientation
    vol_image - a 3D numpy array representing the volume image (helper object)
    offset - the offset from the edges where the axes will be drawn (default=3)
    line_width - the width of the axes lines (default=1)
    """
    max0, max1, max2 = vol_image.shape
    # drawing axes,  y axis0 is the longest (the height) and reversed,
    # the rows x axis1 is half and reversed,
    # the z axis2 (columns) is the shortest"
    vol_image[
        :, max1 - offset - line_width : max1 - offset, offset : offset + line_width
    ] = 3
    vol_image[
        max0 - offset - line_width : max0 - offset,  # offset : offset + line_width,
        max1 // 2 :,
        offset : offset + line_width,
    ] = 1
    vol_image[
        max0 - offset - line_width : max0 - offset,  # offset : offset + line_width,
        max1 - offset - line_width : max1 - offset,
        : max2 // 3,
    ] = 2

    return vol_image
