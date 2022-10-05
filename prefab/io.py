import numpy as np
import gdspy
import cv2

def dev2cell(library, device, res, cell_name, layer=1, cntr_type=cv2.CHAIN_APPROX_SIMPLE):
    # flip image (has to be done)
    device = np.flipud(device)

    # get device contours
    ret, thresh = cv2.threshold(device.astype(np.uint8), 0.5, 1, 0)
    contours = cv2.findContours(thresh, cv2.RETR_CCOMP, cntr_type)

    # add polygon(s) to cell
    outers = []
    inners = []
    for idx, contour in enumerate(contours[0]):
        if len(contour) > 2:
            contour = contour/1000 # μm to nm
            points = contour.squeeze().tolist()
            points = list(tuple(sub) for sub in points)
            if contours[1][0][idx][3] == -1:
                outers.append(points)
            else:
                inners.append(points)
    cell = library.new_cell(cell_name)
    poly = gdspy.boolean(outers, inners, 'xor', layer=layer)
    poly = poly.scale(res, res)
    cell.add(poly)
    return gdspy.CellReference(cell)
