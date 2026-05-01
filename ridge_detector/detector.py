import os
import cv2
import imageio.v3 as iio
import skimage as ski
from ridge_detector.constants import *
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from ridge_detector.utils import (LinesUtil, Junction, Crossref, Line, convolve_gauss,
                   bresenham, fix_locations, interpolate_gradient_test,
                   closest_point, normalize_to_half_circle)


class RidgeDetector:
    def __init__(self,
                 line_widths=np.arange(1, 3),
                 low_contrast=100,
                 high_contrast=200,
                 min_len=5,
                 max_len=0,
                 dark_line=True,
                 estimate_width=True,
                 extend_line=False,
                 correct_pos=False
                 ):
        self.low_contrast = low_contrast
        self.high_contrast = high_contrast
        self.min_len = min_len
        self.max_len = max_len
        self.dark_line = dark_line
        self.estimate_width = estimate_width
        self.extend_line = extend_line
        self.correct_pos = correct_pos

        # Calculate sigmas for multiscale detection
        self.sigmas = np.array([lw / (2 * np.sqrt(3)) + 0.5 for lw in line_widths])

        self.clow = self.low_contrast
        self.chigh = self.high_contrast
        if self.dark_line:
            self.clow = 255 - self.high_contrast
            self.chigh = 255 - self.low_contrast

        self.image = None
        self.gray = None
        self.derivatives = None
        self.lower_thresh = None
        self.upper_thresh = None
        self.contours = None
        self.junctions = None
        self.eigvals = None
        self.eigvecs = None
        self.eigval = None
        self.gradx = None
        self.grady = None
        self.sigma_map = None
        self.normx = None
        self.normy = None
        self.posx = None
        self.posy = None
        self.ismax = None
        self.mode = LinesUtil.MODE_DARK if self.dark_line else LinesUtil.MODE_LIGHT

    def apply_filtering(self):
        height, width = self.gray.shape[:2]
        num_scales = len(self.sigmas)
        saliency = np.zeros((height, width, num_scales), dtype=float)
        orientation = np.zeros((height, width, 2, num_scales), dtype=float)
        rys = np.zeros((height, width, num_scales), dtype=float)
        rxs = np.zeros((height, width, num_scales), dtype=float)
        ryys = np.zeros((height, width, num_scales), dtype=float)
        rxys = np.zeros((height, width, num_scales), dtype=float)
        rxxs = np.zeros((height, width, num_scales), dtype=float)
        symmetric_image = np.zeros((height, width, 2, 2), dtype=float)

        low_threshs = np.zeros((height, width, num_scales), dtype=float)
        high_threshs = np.zeros((height, width, num_scales), dtype=float)
        sigma_maps = np.zeros((height, width, num_scales), dtype=float)

        # Filtering at different scales
        gray = self.gray.astype(float)
        for scale_idx, sigma in enumerate(self.sigmas):
            ry = convolve_gauss(gray, sigma, LinesUtil.DERIV_R)
            rx = convolve_gauss(gray, sigma, LinesUtil.DERIV_C)
            ryy = convolve_gauss(gray, sigma, LinesUtil.DERIV_RR)
            rxy = convolve_gauss(gray, sigma, LinesUtil.DERIV_RC)
            rxx = convolve_gauss(gray, sigma, LinesUtil.DERIV_CC)

            symmetric_image[..., 0, 0] = ryy
            symmetric_image[..., 0, 1] = rxy
            symmetric_image[..., 1, 0] = rxy
            symmetric_image[..., 1, 1] = rxx
            eigvals, eigvecs = np.linalg.eigh(symmetric_image)

            # Maximum absolute eigen as the saliency of lines
            idx = np.absolute(eigvals).argsort()[..., ::-1]
            eigvals_tmp = np.take_along_axis(eigvals, idx, axis=-1)
            eigvecs_tmp = np.take_along_axis(eigvecs, idx[:, :, None, :], axis=-1)

            saliency[:, :, scale_idx] = sigma ** 2.0 * eigvals_tmp[:, :, 0]
            orientation[:, :, :, scale_idx] = eigvecs_tmp[:, :, :, 0]

            # Store intermediate results
            rys[..., scale_idx] = ry
            rxs[..., scale_idx] = rx
            ryys[..., scale_idx] = ryy
            rxys[..., scale_idx] = rxy
            rxxs[..., scale_idx] = rxx

            # Calculate thresholds for each scale using gamma-normalized measurement (gamma is to 2.0)
            line_width = 2 * np.sqrt(3) * (sigma - 0.5)
            low_thresh = (0.17 * sigma ** 2.0 *
                          np.floor(self.clow * line_width / (np.sqrt(2 * np.pi) * sigma ** 3) *
                                   np.exp(-line_width ** 2 / (8 * sigma ** 2))))
            high_thresh = (0.17 * sigma ** 2.0 *
                           np.floor(self.chigh * line_width / (np.sqrt(2 * np.pi) * sigma ** 3) *
                                    np.exp(-line_width ** 2 / (8 * sigma ** 2))))
            low_threshs[..., scale_idx] = low_thresh
            high_threshs[..., scale_idx] = high_thresh
            sigma_maps[..., scale_idx] = sigma

        # Get the scale index of the maximum saliency and the corresponding derivatives and thresholds
        global_max_idx = saliency.argsort()[..., -1]
        self.lower_thresh = np.squeeze(np.take_along_axis(low_threshs, global_max_idx[:, :, None], axis=-1))
        self.upper_thresh = np.squeeze(np.take_along_axis(high_threshs, global_max_idx[:, :, None], axis=-1))
        self.sigma_map = np.squeeze(np.take_along_axis(sigma_maps, global_max_idx[:, :, None], axis=-1))

        self.derivatives = np.zeros((5, height, width), dtype=float)
        self.derivatives[0, ...] = np.squeeze(np.take_along_axis(rys, global_max_idx[:, :, None], axis=-1))
        self.derivatives[1, ...] = np.squeeze(np.take_along_axis(rxs, global_max_idx[:, :, None], axis=-1))
        self.derivatives[2, ...] = np.squeeze(np.take_along_axis(ryys, global_max_idx[:, :, None], axis=-1))
        self.derivatives[3, ...] = np.squeeze(np.take_along_axis(rxys, global_max_idx[:, :, None], axis=-1))
        self.derivatives[4, ...] = np.squeeze(np.take_along_axis(rxxs, global_max_idx[:, :, None], axis=-1))

        self.grady = self.derivatives[0, ...]
        self.gradx = self.derivatives[1, ...]

        self.eigvals = np.take_along_axis(saliency, global_max_idx[:, :, None], axis=-1)
        self.eigvecs = np.take_along_axis(orientation, global_max_idx[:, :, None, None], axis=-1)

    def compute_line_points(self):
        height, width = self.gray.shape[:2]
        self.ismax = np.zeros((height, width), dtype=int)
        self.eigval = np.zeros((height, width), dtype=float)
        self.normx = np.zeros((height, width), dtype=float)
        self.normy = np.zeros((height, width), dtype=float)
        self.posx = np.zeros((height, width), dtype=float)
        self.posy = np.zeros((height, width), dtype=float)

        ry = self.derivatives[0, ...]
        rx = self.derivatives[1, ...]
        ryy = self.derivatives[2, ...]
        rxy = self.derivatives[3, ...]
        rxx = self.derivatives[4, ...]

        val = self.eigvals[:, :, 0] if self.mode == LinesUtil.MODE_DARK else -self.eigvals[:, :, 0]
        val_mask = val > 0.0
        self.eigval[val_mask] = val[val_mask]

        # Equations (22) and (23) in
        # Steger, C. (1998). An unbiased detector of curvilinear structures. (DOI: 10.1109/34.659930)
        nx_ = self.eigvecs[..., 1, 0]
        ny_ = self.eigvecs[..., 0, 0]
        numerator = ry * ny_ + rx * nx_
        denominator = ryy * ny_ ** 2 + 2.0 * rxy * nx_ * ny_ + rxx * nx_ ** 2

        # The minus sign in Eq. (23) is ignored to follow the logic of detecting black ridges in white background
        t = numerator / (denominator + np.finfo(float).eps)
        py_ = t * ny_
        px_ = t * nx_

        bnd_mask = (abs(py_) <= PIXEL_BOUNDARY) & (abs(px_) <= PIXEL_BOUNDARY)
        base_mask = val_mask & bnd_mask
        upper_mask = base_mask & (val >= self.upper_thresh)
        lower_mask = base_mask & (val >= self.lower_thresh) & (val < self.upper_thresh)
        self.ismax[upper_mask] = 2
        self.ismax[lower_mask] = 1

        self.normy[base_mask] = ny_[base_mask]
        self.normx[base_mask] = nx_[base_mask]
        Y, X = np.mgrid[0:height, 0:width]
        self.posy[base_mask] = Y[base_mask] + py_[base_mask]
        self.posx[base_mask] = X[base_mask] + px_[base_mask]

    def extend_lines(self, label):
        height, width = label.shape[:2]
        num_junc = len(self.junctions)
        s = 1 if self.mode == LinesUtil.MODE_DARK else -1
        length = 2.5 * self.sigma_map
        max_line = np.ceil(length * 1.2).astype(int)
        num_cont = len(self.contours)
        for idx_cont in range(num_cont):
            tmp_cont = self.contours[idx_cont]
            num_pnt = tmp_cont.num
            if num_pnt == 1 or tmp_cont.get_contour_class() == LinesUtil.ContourClass.cont_closed:
                continue

            # Check both ends of the line (it==-1: start, it==1: end).
            for it in [-1, 1]:
                if it == -1:
                    trow = tmp_cont.row
                    tcol = tmp_cont.col
                    tangle = tmp_cont.angle
                    tresp = tmp_cont.response

                    # Start point of the line.
                    if tmp_cont.get_contour_class() in [LinesUtil.ContourClass.cont_start_junc,
                                                        LinesUtil.ContourClass.cont_both_junc]:
                        continue
                    dy, dx = trow[1] - trow[0], tcol[1] - tcol[0]
                    alpha = tangle[0]
                    ny, nx = np.sin(alpha), np.cos(alpha)
                    if ny * dx - nx * dy < 0:
                        # Turn the normal by +90 degrees.
                        my, mx = -nx, ny
                    else:
                        # Turn the normal by -90 degrees.
                        my, mx = nx, -ny
                    py, px = trow[0], tcol[0]
                    response = tresp[0]
                else:
                    trow = tmp_cont.row
                    tcol = tmp_cont.col
                    tangle = tmp_cont.angle
                    tresp = tmp_cont.response

                    # End point of the line.
                    if tmp_cont.get_contour_class() in [LinesUtil.ContourClass.cont_end_junc,
                                                        LinesUtil.ContourClass.cont_both_junc]:
                        continue
                    dy, dx = trow[num_pnt - 1] - trow[num_pnt - 2], tcol[num_pnt - 1] - tcol[num_pnt - 2]
                    alpha = tangle[num_pnt - 1]
                    ny, nx = np.sin(alpha), np.cos(alpha)
                    if ny * dx - nx * dy < 0:
                        # Turn the normal by -90 degrees.
                        my, mx = nx, -ny
                    else:
                        # Turn the normal by +90 degrees.
                        my, mx = -nx, ny
                    py, px = trow[num_pnt - 1], tcol[num_pnt - 1]
                    response = tresp[num_pnt - 1]

                # Determine the current pixel and calculate the pixels on the search line.
                y, x = int(py + 0.5), int(px + 0.5)
                dy, dx = py - y, px - x
                line = bresenham(my, mx, max_line[LinesUtil.BR(y, height), LinesUtil.BC(x, width)], dy, dx)
                num_line = line.shape[0]
                exty, extx = np.zeros(num_line, dtype=int), np.zeros(num_line, dtype=int)

                # Now determine whether we can go only uphill (bright lines)
                # or downhill (dark lines) until we hit another line.
                num_add = 0
                add_ext = False
                for k in range(num_line):
                    nexty, nextx = y + line[k, 0], x + line[k, 1]
                    nextpy, nextpx, t = closest_point(py, px, my, mx, nexty, nextx)

                    # Ignore points before or less than half a pixel away from the true end point of the line.
                    if t <= 0.5:
                        continue
                    # Stop if the gradient can't be interpolated any more or if the next point lies outside the image.
                    if (nextpx < 0 or nextpy < 0 or nextpy >= height - 1 or nextpx >= width - 1 or
                            nextx < 0 or nexty < 0 or nexty >= height or nextx >= width):
                        break
                    gy, gx = interpolate_gradient_test(self.grady, self.gradx, nextpy, nextpx)

                    # Stop if we can't go uphill anymore.
                    # This is determined by the dot product of the line direction and the gradient.
                    # If it is smaller than 0 we go downhill (reverse for dark lines).
                    if s * (mx * gx + my * gy) < 0 and label[nexty, nextx] == 0:
                        break
                    # Have we hit another line?
                    if label[nexty, nextx] > 0:
                        m = label[nexty, nextx] - 1
                        # Search for the junction point on the other line.
                        dist = np.sqrt((nextpy - self.contours[m].row) ** 2 + (nextpx - self.contours[m].col) ** 2)
                        j = np.argmin(dist)

                        exty[num_add] = self.contours[m].row[j]
                        extx[num_add] = self.contours[m].col[j]
                        end_resp = self.contours[m].response[j]
                        end_angle = self.contours[m].angle[j]
                        beta = end_angle
                        if beta >= np.pi:
                            beta -= np.pi
                        diff1 = abs(beta - alpha)
                        if diff1 >= np.pi:
                            diff1 = 2.0 * np.pi - diff1
                        diff2 = abs(beta + np.pi - alpha)
                        if diff2 >= np.pi:
                            diff2 = 2.0 * np.pi - diff2
                        if diff1 < diff2:
                            end_angle = beta
                        else:
                            end_angle = beta + np.pi
                        num_add += 1
                        add_ext = True
                        break
                    else:
                        exty[num_add], extx[num_add] = nextpy, nextpx
                        num_add += 1

                if add_ext:
                    # Make room for the new points.
                    num_pnt += num_add
                    new_row = np.zeros(num_pnt, dtype=float)
                    new_col = np.zeros(num_pnt, dtype=float)
                    new_angle = np.zeros(num_pnt, dtype=float)
                    new_resp = np.zeros(num_pnt, dtype=float)

                    tmp_cont.row = new_row
                    tmp_cont.col = new_col
                    tmp_cont.angle = new_angle
                    tmp_cont.response = new_resp
                    tmp_cont.num = num_pnt
                    if it == -1:
                        tmp_cont.row[num_add:] = trow
                        tmp_cont.row[:num_add] = exty[:num_add][::-1]
                        tmp_cont.col[num_add:] = tcol
                        tmp_cont.col[:num_add] = extx[:num_add][::-1]
                        tmp_cont.angle[num_add:] = tangle
                        tmp_cont.angle[:num_add] = float(alpha)
                        tmp_cont.response[num_add:] = tresp
                        tmp_cont.response[:num_add] = float(response)
                        tmp_cont.angle[0] = end_angle
                        tmp_cont.response[0] = end_resp
                        # Adapt indices of the previously found junctions.
                        for k in range(num_junc):
                            if self.junctions[k].cont1 == idx_cont:
                                self.junctions[k].pos += num_add
                    else:
                        # Insert points at the end of the line.
                        tmp_cont.row[:num_pnt - num_add] = trow
                        tmp_cont.row[num_pnt - num_add:] = exty[:num_add]
                        tmp_cont.col[:num_pnt - num_add] = tcol
                        tmp_cont.col[num_pnt - num_add:] = extx[:num_add]
                        tmp_cont.angle[:num_pnt - num_add] = tangle
                        tmp_cont.angle[num_pnt - num_add:] = float(alpha)
                        tmp_cont.response[:num_pnt - num_add] = tresp
                        tmp_cont.response[num_pnt - num_add:] = float(response)
                        tmp_cont.angle[-1] = end_angle
                        tmp_cont.response[-1] = end_resp

                    # Add the junction point only if it is not one of the other line's endpoints.
                    if 0 < j < self.contours[m].num - 1:
                        if it == -1:
                            if tmp_cont.get_contour_class() == LinesUtil.ContourClass.cont_end_junc:
                                tmp_cont.set_contour_class(LinesUtil.ContourClass.cont_both_junc)
                            else:
                                tmp_cont.set_contour_class(LinesUtil.ContourClass.cont_start_junc)
                        else:
                            if tmp_cont.get_contour_class() == LinesUtil.ContourClass.cont_start_junc:
                                tmp_cont.set_contour_class(LinesUtil.ContourClass.cont_both_junc)
                            else:
                                tmp_cont.set_contour_class(LinesUtil.ContourClass.cont_end_junc)

                        if it == -1:
                            self.junctions.append(Junction(m, idx_cont, j, tmp_cont.row[0], tmp_cont.col[0]))
                        else:
                            self.junctions.append(
                                Junction(m, idx_cont, j, tmp_cont.row[num_pnt - 1], tmp_cont.col[num_pnt - 1]))

                        num_junc += 1

    def compute_contours(self):
        height, width = self.eigval.shape[:2]
        label = np.zeros((height, width), dtype=int)
        indx = np.zeros((height, width), dtype=int)

        num_cont, num_junc = 0, 0
        self.junctions, self.contours = [], []

        cross = []
        area = 0
        for r_idx in range(height):
            for c_idx in range(width):
                if self.ismax[r_idx, c_idx] >= 2:
                    area += 1
                    cross.append(Crossref(r_idx, c_idx, self.eigval[r_idx, c_idx], False))

        response_2d = self.eigval.reshape(height, width)
        resp_dr = convolve(response_2d, kernel_r, mode='mirror')
        resp_dc = convolve(response_2d, kernel_c, mode='mirror')
        resp_dd = convolve(response_2d, kernel_d, mode='mirror')
        resp_drr = convolve(response_2d, kernel_rr, mode='mirror')
        resp_drc = convolve(response_2d, kernel_rc, mode='mirror')
        resp_dcc = convolve(response_2d, kernel_cc, mode='mirror')

        # Sorting cross list in ascending order by value
        cross.sort()

        # Updating indx based on the sorted cross list
        for ci, cref in enumerate(cross):
            indx[cref.y, cref.x] = ci + 1

        indx_max = 0
        while True:
            cls = LinesUtil.ContourClass.cont_no_junc
            while indx_max < area and cross[indx_max].done:
                indx_max += 1

            if indx_max == area:
                break

            max_val = cross[indx_max].value
            maxy, maxx = cross[indx_max].y, cross[indx_max].x
            if max_val == 0.0:
                break

            # Initialize line data
            row, col, angle, resp = [], [], [], []

            # Add starting point to the line.
            num_pnt = 0
            label[maxy, maxx] = num_cont + 1
            if indx[maxy, maxx] != 0:
                cross[indx[maxy, maxx] - 1].done = True

            # Select line direction
            row.append(maxy)
            col.append(maxx)
            nx = -self.normx[maxy, maxx]
            ny = self.normy[maxy, maxx]
            alpha = normalize_to_half_circle(np.arctan2(ny, nx))
            octant = int(np.floor(4.0 / np.pi * alpha + 0.5)) % 4

            ''' * Select normal to the line. The normal points to the right of the line as the
                * line is traversed from 0 to num-1. Since the points are sorted in reverse
                * order before the second iteration, the first beta actually has to point to
                * the left of the line!
            '''

            beta = alpha + np.pi / 2.0
            if beta >= 2.0 * np.pi:
                beta -= 2.0 * np.pi
            angle.append(beta)
            yy = self.posy[maxy, maxx] - maxy
            xx = self.posx[maxy, maxx] - maxx
            interpolated_response = (resp_dd[maxy, maxx] + yy * resp_dr[maxy, maxx] + xx * resp_dc[maxy, maxx] +
                                     yy ** 2 * resp_drr[maxy, maxx] + xx * yy * resp_drc[maxy, maxx] +
                                     xx ** 2 * resp_dcc[maxy, maxx])
            resp.append(interpolated_response)
            num_pnt += 1

            # Mark double responses as processed.
            for ni in range(2):
                nexty = maxy + cleartab[octant][ni][0]
                nextx = maxx + cleartab[octant][ni][1]
                if nexty < 0 or nexty >= height or nextx < 0 or nextx >= width:
                    continue
                if self.ismax[nexty, nextx] > 0:
                    nx = -self.normx[nexty, nextx]
                    ny = self.normy[nexty, nextx]
                    nextalpha = normalize_to_half_circle(np.arctan2(ny, nx))
                    diff = abs(alpha - nextalpha)
                    if diff >= np.pi / 2.0:
                        diff = np.pi - diff
                    if diff < MAX_ANGLE_DIFFERENCE:
                        label[nexty, nextx] = num_cont + 1
                        if indx[nexty, nextx] != 0:
                            cross[indx[nexty, nextx] - 1].done = True

            for it in range(1, 3):
                y, x = maxy, maxx
                ny, nx = self.normy[y, x], -self.normx[y, x]

                alpha = normalize_to_half_circle(np.arctan2(ny, nx))
                last_octant = int(np.floor(4.0 / np.pi * alpha + 0.5)) % 4 if it == 1 else \
                    int(np.floor(4.0 / np.pi * alpha + 0.5)) % 4 + 4
                last_beta = alpha + np.pi / 2.0
                if last_beta >= 2.0 * np.pi:
                    last_beta -= 2.0 * np.pi

                if it == 2:
                    # Sort the points found in the first iteration in reverse.
                    row.reverse()
                    col.reverse()
                    angle.reverse()
                    resp.reverse()

                while True:
                    ny, nx = self.normy[y, x], -self.normx[y, x]
                    py, px = self.posy[y, x], self.posx[y, x]

                    # Orient line direction with respect to the last line direction
                    alpha = normalize_to_half_circle(np.arctan2(ny, nx))
                    octant = int(np.floor(4.0 / np.pi * alpha + 0.5)) % 4

                    if octant == 0 and 3 <= last_octant <= 5:
                        octant = 4
                    elif octant == 1 and 4 <= last_octant <= 6:
                        octant = 5
                    elif octant == 2 and 4 <= last_octant <= 7:
                        octant = 6
                    elif octant == 3 and (last_octant == 0 or last_octant >= 6):
                        octant = 7
                    last_octant = octant

                    # Determine appropriate neighbor
                    nextismax = False
                    nexti = 1
                    mindiff = float('inf')
                    for ti in range(3):
                        nexty, nextx = y + dirtab[octant][ti][0], x + dirtab[octant][ti][1]
                        if nexty < 0 or nexty >= height or nextx < 0 or nextx >= width:
                            continue
                        if self.ismax[nexty, nextx] == 0:
                            continue
                        nextpy, nextpx = self.posy[nexty, nextx], self.posx[nexty, nextx]
                        dy, dx = nextpy - py, nextpx - px
                        dist = np.sqrt(dx ** 2 + dy ** 2)
                        ny, nx = self.normy[nexty, nextx], -self.normx[nexty, nextx]
                        nextalpha = normalize_to_half_circle(np.arctan2(ny, nx))
                        diff = abs(alpha - nextalpha)
                        if diff >= np.pi / 2.0:
                            diff = np.pi - diff
                        diff = dist + diff
                        if diff < mindiff:
                            mindiff = diff
                            nexti = ti
                        if not (self.ismax[nexty, nextx] == 0):
                            nextismax = True

                    # Mark double responses as processed
                    for ni in range(2):
                        nexty, nextx = y + cleartab[octant][ni][0], x + cleartab[octant][ni][1]
                        if nexty < 0 or nexty >= height or nextx < 0 or nextx >= width:
                            continue
                        if self.ismax[nexty, nextx] > 0:
                            ny, nx = self.normy[nexty, nextx], -self.normx[nexty, nextx]
                            nextalpha = normalize_to_half_circle(np.arctan2(ny, nx))
                            diff = abs(alpha - nextalpha)
                            if diff >= np.pi / 2.0:
                                diff = np.pi - diff
                            if diff < MAX_ANGLE_DIFFERENCE:
                                label[nexty, nextx] = num_cont + 1
                                if not (indx[nexty, nextx] == 0):
                                    cross[indx[nexty, nextx] - 1].done = True

                    # Have we found the end of the line?
                    if not nextismax:
                        break  # Exit the loop if the end of the line is found

                    # Add the neighbor to the line if not at the end
                    y += dirtab[octant][nexti][0]
                    x += dirtab[octant][nexti][1]

                    row.append(self.posy[y, x])
                    col.append(self.posx[y, x])
                    # Orient normal to the line direction with respect to the last normal
                    ny, nx = self.normy[y, x], self.normx[y, x]

                    beta = normalize_to_half_circle(np.arctan2(ny, nx))
                    diff1 = min(abs(beta - last_beta), 2.0 * np.pi - abs(beta - last_beta))
                    alt_beta = (beta + np.pi) % (2.0 * np.pi)  # Normalize alternative beta
                    diff2 = min(abs(alt_beta - last_beta), 2.0 * np.pi - abs(alt_beta - last_beta))
                    # Choose the angle with the smallest difference and update
                    chosen_beta = beta if diff1 < diff2 else alt_beta
                    angle.append(chosen_beta)
                    last_beta = chosen_beta

                    yy, xx = self.posy[y, x] - maxy, self.posx[y, x] - maxx
                    interpolated_response = (resp_dd[y, x] + yy * resp_dr[y, x] + xx * resp_dc[y, x] +
                                             yy ** 2 * resp_drr[y, x] + xx * yy * resp_drc[y, x] +
                                             xx ** 2 * resp_dcc[y, x])
                    resp.append(interpolated_response)
                    num_pnt += 1

                    # If the appropriate neighbor is already processed a junction point is found
                    if label[y, x] > 0:
                        k = label[y, x] - 1
                        if k == num_cont:
                            # Line intersects itself
                            for j in range(num_pnt - 1):
                                if row[j] == self.posy[y, x] and col[j] == self.posx[y, x]:
                                    if j == 0:
                                        # Contour is closed
                                        cls = LinesUtil.ContourClass.cont_closed
                                        row.reverse()
                                        col.reverse()
                                        angle.reverse()
                                        resp.reverse()
                                        it = 2
                                    else:
                                        # Determine contour class
                                        if it == 2:
                                            if cls == LinesUtil.ContourClass.cont_start_junc:
                                                cls = LinesUtil.ContourClass.cont_both_junc
                                            else:
                                                cls = LinesUtil.ContourClass.cont_end_junc
                                            # Index j is correct
                                            self.junctions.append(
                                                Junction(num_cont, num_cont, j, self.posy[y, x], self.posx[y, x]))
                                            num_junc += 1
                                        else:
                                            cls = LinesUtil.ContourClass.cont_start_junc
                                            # Index num_pnt-1-j is correct since the line will be sorted in reverse
                                            self.junctions.append(Junction(num_cont, num_cont,
                                                                      num_pnt - 1 - j, self.posy[y, x],
                                                                      self.posx[y, x]))
                                            num_junc += 1
                                    break
                            j = -1
                        else:
                            for j in range(self.contours[k].num):
                                if self.contours[k].row[j] == self.posy[y, x] and self.contours[k].col[j] == self.posx[y, x]:
                                    break
                            if j == self.contours[k].num:
                                # No point found on the other line, a double response occurred
                                dist = np.sqrt(
                                    (self.posy[y, x] - self.contours[k].row) ** 2 + (self.posx[y, x] - self.contours[k].col) ** 2)
                                j = np.argmin(dist)
                                row.append(self.contours[k].row[j])
                                col.append(self.contours[k].col[j])
                                beta = self.contours[k].angle[j]
                                if beta >= np.pi:
                                    beta -= np.pi
                                diff1 = abs(beta - last_beta)
                                if diff1 >= np.pi:
                                    diff1 = 2.0 * np.pi - diff1
                                diff2 = abs(beta + np.pi - last_beta)
                                if diff2 >= np.pi:
                                    diff2 = 2.0 * np.pi - diff2
                                angle.append(beta) if diff1 < diff2 else angle.append(beta + np.pi)
                                resp.append(self.contours[k].response[j])
                                num_pnt += 1
                        if 0 < j < self.contours[k].num - 1:
                            # Determine contour class
                            if it == 1:
                                cls = LinesUtil.ContourClass.cont_start_junc
                            elif cls == LinesUtil.ContourClass.cont_start_junc:
                                cls = LinesUtil.ContourClass.cont_both_junc
                            else:
                                cls = LinesUtil.ContourClass.cont_end_junc

                            # Add the new junction
                            self.junctions.append(Junction(k, num_cont, j, row[num_pnt - 1], col[num_pnt - 1]))
                            num_junc += 1
                        break

                    label[y, x] = num_cont + 1
                    if indx[y, x] != 0:
                        cross[indx[y, x] - 1].done = True

            if num_pnt > 1:
                # Create a new Line object and copy the current line's attributes
                new_line = Line()
                new_line.row = np.array(row)
                new_line.col = np.array(col)
                new_line.angle = np.array(angle)
                new_line.response = np.array(resp)
                new_line.width_r = None
                new_line.width_l = None
                new_line.asymmetry = None
                new_line.intensity = None
                new_line.num = num_pnt
                new_line.set_contour_class(cls)

                # Add the new line to the list of contours
                self.contours.append(new_line)
                num_cont += 1
            else:
                # Delete the point from the label image; using maxx and maxy as coordinates in the label image
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if label[LinesUtil.BR(maxy + i, height), LinesUtil.BC(maxx + j, width)] == num_cont + 1:
                            label[LinesUtil.BR(maxy + i, height), LinesUtil.BC(maxx + j, width)] = 0

        if self.extend_line:
            self.extend_lines(label)

        # Adjust angles to point to the right of the line
        for i in range(num_cont):
            tmp_cont = self.contours[i]
            num_pnt = tmp_cont.num
            if num_pnt > 1:
                k = (num_pnt - 1) // 2
                dy, dx = tmp_cont.row[k + 1] - tmp_cont.row[k], tmp_cont.col[k + 1] - tmp_cont.col[k]
                ny, nx = np.sin(tmp_cont.angle[k]), np.cos(tmp_cont.angle[k])

                # If angles point to the left of the line, they have to be adapted
                if ny * dx - nx * dy < 0:
                    tmp_cont.angle = np.array([(ang + np.pi) % (2 * np.pi) for ang in tmp_cont.angle])

    def compute_line_width(self):
        height, width = self.grady.shape[:2]
        length = 2.5 * self.sigma_map
        max_length = np.ceil(length * 1.2).astype(int)
        grad = np.sqrt(self.grady ** 2 + self.gradx ** 2)

        grad_dr = convolve(grad, kernel_r, mode='mirror')
        grad_dc = convolve(grad, kernel_c, mode='mirror')
        grad_dd = convolve(grad, kernel_d, mode='mirror')
        grad_drr = convolve(grad, kernel_rr, mode='mirror')
        grad_drc = convolve(grad, kernel_rc, mode='mirror')
        grad_dcc = convolve(grad, kernel_cc, mode='mirror')

        symmetric_image = np.zeros((height, width, 2, 2), dtype=float)
        symmetric_image[..., 0, 0] = 2 * grad_drr
        symmetric_image[..., 0, 1] = grad_drc
        symmetric_image[..., 1, 0] = grad_drc
        symmetric_image[..., 1, 1] = 2 * grad_dcc
        eigvals, eigvecs = np.linalg.eigh(symmetric_image)
        idx = np.absolute(eigvals).argsort()[..., ::-1]
        eigvals = np.take_along_axis(eigvals, idx, axis=-1)
        eigvecs = np.take_along_axis(eigvecs, idx[:, :, None, :], axis=-1)

        bb = grad_dr * eigvecs[:, :, 0, 0] + grad_dc * eigvecs[:, :, 1, 0]
        aa = 2.0 * (grad_drr * eigvecs[:, :, 0, 0] ** 2 + grad_drc * eigvecs[:, :, 0, 0] * eigvecs[:, :, 1, 0] +
                    grad_dcc * eigvecs[:, :, 1, 0] ** 2)
        tt = bb / (aa + np.finfo(float).eps)
        pp1, pp2 = tt * eigvecs[:, :, 0, 0], tt * eigvecs[:, :, 1, 0]
        grad_rl = (grad_dd + pp1 * grad_dr + pp2 * grad_dc +
                   pp1 * pp1 * grad_drr + pp1 * pp2 * grad_drc + pp2 * pp2 * grad_dcc)

        for i, cont in enumerate(self.contours):
            num_points = cont.num
            width_l = np.zeros(num_points, dtype=float)
            width_r = np.zeros(num_points, dtype=float)
            grad_l = np.zeros(num_points, dtype=float)
            grad_r = np.zeros(num_points, dtype=float)
            pos_x = np.zeros(num_points, dtype=float)
            pos_y = np.zeros(num_points, dtype=float)

            for j in range(num_points):
                py, px = cont.row[j], cont.col[j]
                pos_y[j], pos_x[j] = py, px
                r, c = LinesUtil.BR(round(py), height), LinesUtil.BC(round(px), width)
                ny, nx = np.sin(cont.angle[j]), np.cos(cont.angle[j])

                line = bresenham(ny, nx, max_length[r, c])
                num_line = line.shape[0]
                width_r[j] = width_l[j] = 0

                for direct in [-1, 1]:
                    for k in range(num_line):
                        y, x = LinesUtil.BR(r + direct * line[k, 0], height), LinesUtil.BC(c + direct * line[k, 1], width)
                        val = -eigvals[y, x, 0]
                        if val > 0.0:
                            p1, p2 = pp1[y, x], pp2[y, x]

                            if abs(p1) <= 0.5 and abs(p2) <= 0.5:
                                t = ny * (py - (r + direct * line[k, 0] + p1)) + nx * (px - (c + direct * line[k, 1] + p2))
                                if direct == 1:
                                    grad_r[j] = grad_rl[y, x]
                                    width_r[j] = abs(t)
                                else:
                                    grad_l[j] = grad_rl[y, x]
                                    width_l[j] = abs(t)
                                break
            fix_locations(cont, width_l, width_r, grad_l, grad_r, pos_y, pos_x,
                          self.sigma_map, self.correct_pos, self.mode)

    def prune_contours(self):
        if self.min_len <= 0:
            return

        id_remove = []
        conts = []
        for i in range(len(self.contours)):
            cont_len = self.contours[i].estimate_length()
            if cont_len < self.min_len or (0 < self.max_len < cont_len):
                id_remove.append(self.contours[i].id)
            else:
                conts.append(self.contours[i])

        juncs = []
        for junc in self.junctions:
            if junc.cont1 not in id_remove and junc.cont2 not in id_remove:
                juncs.append(junc)

        self.contours = conts
        self.junctions = juncs

    def detect_lines(self, image):
        self.image = iio.imread(image) if isinstance(image, str) else image

        # Normalize to uint8 if needed
        if self.image.dtype != np.uint8:
            img = self.image.astype(float)
            self.image = ((img - img.min()) / (img.max() - img.min() + 1e-12) * 255).astype(np.uint8)

        # Convert to grayscale
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY) if self.image.ndim == 3 else self.image

        self.apply_filtering()
        self.compute_line_points()
        self.compute_contours()
        if self.estimate_width:
            self.compute_line_width()
        self.prune_contours()

    def save_results(self, save_dir=None, prefix="", make_binary=True, draw_junc=False, draw_width=True):
        all_contour_points, all_width_left, all_width_right = [], [], []
        for cont in self.contours:
            num_points = cont.num
            contour_points, width_left, width_right = [], [], []

            for j in range(num_points):
                px, py = cont.col[j], cont.row[j]
                nx, ny = np.cos(cont.angle[j]), np.sin(cont.angle[j])
                contour_points.append([round(px), round(py)])

                if draw_width and self.estimate_width:
                    px_r, py_r = px + cont.width_r[j] * nx, py + cont.width_r[j] * ny
                    px_l, py_l = px - cont.width_l[j] * nx, py - cont.width_l[j] * ny

                    # if last_w_r > 0 and cont.width_r[j] > 0:
                    width_right.append([round(px_r), round(py_r)])

                    # if last_w_l > 0 and cont.width_l[j] > 0:
                    width_left.append([round(px_l), round(py_l)])

            all_contour_points.append(np.array(contour_points))
            if draw_width and self.estimate_width:
                all_width_right.append(np.array(width_right))
                all_width_left.append(np.array(width_left))

        if save_dir is None:
            save_dir = os.getcwd()

        result_img = self.image.copy() if self.image.ndim > 2 else np.repeat(self.image[:, :, None], 3, axis=2)

        img = cv2.polylines(result_img, all_contour_points, False, (255, 0, 0))
        iio.imwrite(os.path.join(save_dir, f"{prefix}_contours.png"), img)

        if draw_width and self.estimate_width:
            img = cv2.polylines(img, all_width_right, False, (0, 255, 0))
            img = cv2.polylines(img, all_width_left, False, (0, 255, 0))
            iio.imwrite(os.path.join(save_dir, f"{prefix}_contours_widths.png"), img)

        if draw_junc:
            for junc in self.junctions:
                img = cv2.circle(img, (round(junc.x), round(junc.y)), 2, (0, 255, 255), -1)
            iio.imwrite(os.path.join(save_dir, f"{prefix}_contours_widths_junctions.png"), img)

        if make_binary:
            height, width = self.image.shape[:2]
            binary_contours = np.ones((height, width), dtype=np.uint8) * 255

            for contour_points in all_contour_points:
                for points in contour_points:
                    binary_contours[min([points[1], height-1]), min([points[0], width-1])] = 0
            iio.imwrite(os.path.join(save_dir, f"{prefix}_binary_contours.png"), binary_contours)

            if draw_width and self.estimate_width:
                binary_width = np.ones((height, width), dtype=np.uint8) * 255
                for width_left, width_right in zip(all_width_left, all_width_right):
                    poly_points = np.concatenate((width_left, width_right[::-1, :]), axis=0)
                    mask = ski.draw.polygon2mask((height, width), poly_points[:, [1, 0]])
                    binary_width[mask] = 0
                iio.imwrite(os.path.join(save_dir, f"{prefix}_binary_widths.png"), binary_width)

    def show_results(self):
        all_contour_points, all_width_left, all_width_right = [], [], []
        height, width = self.image.shape[:2]
        for cont in self.contours:
            num_points = cont.num
            last_w_r, last_w_l = 0, 0
            contour_points, width_left, width_right = [], [], []

            for j in range(num_points):
                px, py = cont.col[j], cont.row[j]
                nx, ny = np.cos(cont.angle[j]), np.sin(cont.angle[j])
                contour_points.append([LinesUtil.BC(round(px), width), LinesUtil.BR(round(py), height)])

                if self.estimate_width:
                    px_r, py_r = px + cont.width_r[j] * nx, py + cont.width_r[j] * ny
                    px_l, py_l = px - cont.width_l[j] * nx, py - cont.width_l[j] * ny

                    if last_w_r > 0 and cont.width_r[j] > 0:
                        width_right.append([round(px_r), round(py_r)])

                    if last_w_l > 0 and cont.width_l[j] > 0:
                        width_left.append([round(px_l), round(py_l)])

                    last_w_r, last_w_l = cont.width_r[j], cont.width_l[j]

            all_contour_points.append(np.array(contour_points))

            if self.estimate_width:
                all_width_right.append(np.array(width_right))
                all_width_left.append(np.array(width_left))

        result_img = self.image.copy() if self.image.ndim > 2 else np.repeat(self.image[:, :, None], 3, axis=2)
        img_contours = cv2.polylines(result_img, all_contour_points, False, (255, 0, 0))
        img_cont_width = cv2.polylines(img_contours.copy(), all_width_right, False, (0, 255, 0))
        img_cont_width = cv2.polylines(img_cont_width, all_width_left, False, (0, 255, 0))

        height, width = self.image.shape[:2]
        binary_contours = np.ones((height, width), dtype=np.uint8) * 255

        for contour_points in all_contour_points:
            for points in contour_points:
                binary_contours[min([points[1], height - 1]), min([points[0], width - 1])] = 0

        binary_widths = np.ones((height, width), dtype=np.uint8) * 255
        for width_left, width_right in zip(all_width_left, all_width_right):
            if width_left.size == 0 or width_right.size == 0:
                continue
            poly_points = np.concatenate((width_left, width_right[::-1, :]), axis=0)
            mask = ski.draw.polygon2mask((height, width), poly_points[:, [1, 0]])
            binary_widths[mask] = 0

        if self.estimate_width:
            fig, axes = plt.subplots(2, 2, figsize=(16, 16))
            axes[0, 0].imshow(img_contours)
            axes[0, 0].set_title("contours")
            axes[0, 1].imshow(img_cont_width)
            axes[0, 1].set_title("contours and widths")
            axes[1, 0].imshow(binary_contours, cmap='gray')
            axes[1, 0].set_title("binary contours")
            axes[1, 1].imshow(binary_widths, cmap='gray')
            axes[1, 1].set_title("binary widths")
            plt.show()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            axes[0].imshow(img_contours)
            axes[0].set_title("contours")
            axes[1].imshow(binary_contours, cmap='gray')
            axes[1].set_title("binary contours")
            plt.show()


# if __name__ == "__main__":
#     detector = RidgeDetector(line_widths=np.arange(7,11),
#                              low_contrast=50,
#                              high_contrast=100,
#                              min_len=15,
#                              max_len=0,
#                              dark_line=False,
#                              estimate_width=True,
#                              extend_line=True,
#                              correct_pos=False,
#                              )
#
#     detector.detect_lines("../data/images/img2.jpg")
#     detector.show_results()
    # plt.imshow(detector.eigvals)
    # plt.show()
    # detector.save_results("../data/results/", prefix="img7")





