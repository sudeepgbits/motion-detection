"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2
import os


# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to
                             [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """
    
    #cv2.imshow('image a',image)
    #cv2.waitKey(0)
    sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3,scale=1.0/8.0,borderType=cv2.BORDER_DEFAULT)
    print np.max(sobelx)
#    cv2.imshow('image a',sobelx)
#    cv2.waitKey(0)
    return sobelx


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """

    sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3,scale=1.0/8.0,borderType=cv2.BORDER_DEFAULT)
#    cv2.imshow('image sobely',sobely)
#    cv2.waitKey(0)
    return sobely


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """
    
    
    img_a  = cv2.blur(img_a,(3,3))
    img_b  = cv2.blur(img_b,(3,3))
    Ix = gradient_x(img_a)
    Iy = gradient_y(img_a)
    
    
    It = img_b - img_a
    
    
       
    u = np.zeros(img_a.shape)
    v = np.zeros(img_a.shape)
    m,n = img_a.shape
    AtA = np.zeros((m,n,2,2))
    AtA_inv = np.zeros((m,n,2,2))
    Atb = np.zeros((m,n,2,1))
    result = np.zeros((m,n,2,1))
    win = 5
    w = win/2
#    AtA[w:-w,w:-w] =  np.matrix([[np.sum(np.square(Ix[:2*w+1,:2*w+1])), np.sum(np.multiply(Ix[:2*w+1,:2*w+1],Iy[:2*w+1,:2*w+1]))],[np.sum(np.multiply(Ix[:2*w+1,:2*w+1],Iy[:2*w+1,:2*w+1])), np.sum(np.square(Iy[:2*w+1,:2*w+1]))]])
#    
#    Atb[w:-w,w:-w] = -1 * np.matrix([[np.sum(np.multiply(Ix[:2*w+1,:2*w+1],It[:2*w+1,:2*w+1]))],[np.sum(np.multiply(Iy[:2*w+1,:2*w+1],It[:2*w+1,:2*w+1]))]])
#    
    #grad_t = img_b - img_a    

    Ix2 = np.square(Ix)
    Iy2 = np.square(Iy)
    Ixy = np.multiply(Ix,Iy)
    Ixt = np.multiply(Ix,It)
    Iyt = np.multiply(Iy,It)
    
    if k_type == 'uniform':
        #print 'hi'
        win = k_size
        sum_Ix2 = cv2.boxFilter(src=Ix2, ddepth=-1, ksize=(win, win), normalize=False)
        sum_Iy2 = cv2.boxFilter(src=Iy2, ddepth=-1, ksize=(win, win), normalize=False)
        sum_Ixy = cv2.boxFilter(src=Ixy, ddepth=-1, ksize=(win, win), normalize=False)
        sum_Ixt = cv2.boxFilter(src=Ixt, ddepth=-1, ksize=(win, win), normalize=False)
        sum_Iyt = cv2.boxFilter(src=Iyt, ddepth=-1, ksize=(win, win), normalize=False)    
    else:
        kernel = cv2.getGaussianKernel(k_size,sigma, ktype=cv2.CV_64F)
        sum_Ix2 = cv2.sepFilter2D(Ix2, -1, kernel, kernel, borderType = cv2.BORDER_DEFAULT)
        sum_Iy2 = cv2.sepFilter2D(Iy2, -1, kernel, kernel, borderType = cv2.BORDER_DEFAULT)
        sum_Ixy = cv2.sepFilter2D(Ixy, -1, kernel, kernel, borderType = cv2.BORDER_DEFAULT)
        sum_Ixt = cv2.sepFilter2D(Ixt, -1, kernel, kernel, borderType = cv2.BORDER_DEFAULT)
        sum_Iyt = cv2.sepFilter2D(Iyt, -1, kernel, kernel, borderType = cv2.BORDER_DEFAULT)
    #print sum_Ix2[0,0]
    AtA[:,:,0,0] = sum_Ix2
    AtA[:,:,0,1] = sum_Ixy
    AtA[:,:,1,0] = sum_Ixy
    AtA[:,:,1,1] = sum_Iy2
    
    Atb[:,:,0,0] = -1 * sum_Ixt
    Atb[:,:,1,0] = -1 * sum_Iyt
    det_Ata = np.linalg.det(AtA)
    #det_Ata = np.linalg.det(AtA[:,:])
    #AtA_inv[np.where(det_Ata != 0)] = np.linalg.inv(AtA[np.where(det_Ata != 0)])
    #result = np.dot(AtA_inv,Atb)
    for i in range(w, img_a.shape[0]-w):
         for j in range(w, img_a.shape[1]-w):
             if det_Ata[i,j] != 0:
                 AtA_inv[i,j] = np.linalg.inv(AtA[i,j])
                 result[i,j] = np.dot(AtA_inv[i,j],Atb[i,j])
             
#    result = np.linalg.solve(AtA,Atb)
    
    u = result[:,:,0,0]
    v = result[:,:,1,0]
    
    
# =============================================================================
#     for i in range(w, img_a.shape[0]-w):
#         for j in range(w, img_a.shape[1]-w):
#             Ix1 = Ix[i-w:i+w+1, j-w:j+w+1].flatten()
#             Ix1 = Ix1.reshape(len(Ix1),1)
#             Iy1 = Iy[i-w:i+w+1, j-w:j+w+1].flatten()
#             Iy1 = Iy1.reshape(len(Iy1),1)
#             It1 = It[i-w:i+w+1, j-w:j+w+1].flatten()
#             A = np.concatenate((Ix1,Iy1),1)
#             b = It1.reshape(len(It1),1)
#             At = np.transpose(A)
#             AtA = np.dot(At,A)
#             Atb = -1* np.dot(At,b)
#             #print AtA
#             det_Ata = np.linalg.det(AtA)
#             if det_Ata != 0:
#                 #print 'here'
#                 AtA_inv = np.linalg.inv(AtA)
#                 result = np.dot(AtA_inv,Atb)
#                 u[i,j] = result[0]
#                 v[i,j] = result[1]
# =============================================================================
#            AtA[i,j] =  np.matrix([[np.sum(np.square(Ix[i-w:i+w+1, j-w:j+w+1])), np.sum(np.multiply(Ix[i-w:i+w+1, j-w:j+w+1],Iy[i-w:i+w+1, j-w:j+w+1]))],[np.sum(np.multiply(Ix[i-w:i+w+1, j-w:j+w+1],Iy[i-w:i+w+1, j-w:j+w+1])), np.sum(np.square(Iy[i-w:i+w+1, j-w:j+w+1]))]])
#    
#            Atb[i,j] = -1 * np.matrix([[np.sum(np.multiply(Ix[i-w:i+w+1, j-w:j+w+1],It[i-w:i+w+1, j-w:j+w+1]))],[np.sum(np.multiply(Iy[i-w:i+w+1, j-w:j+w+1],It[i-w:i+w+1, j-w:j+w+1]))]])
#    
#            det_Ata = np.linalg.det(AtA[i,j])
#            if det_Ata != 0:
#                AtA_inv[i,j] = np.linalg.inv(AtA[i,j])
#                result = np.dot(AtA_inv[i,j],Atb[i,j])
#                u[i,j] = result[0]
#                v[i,j] = result[1]
    
    
#    result = np.dot(AtA_inv,Atb)
#    u= result[0]
#    v = result[1]
    
    return (u,v)



def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """

    [m,n] = image.shape
    conv_image = np.zeros((m,n))
    
    
    kernel = np.array([1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0])
    kernel = np.outer(kernel,kernel)
    #cv2.imshow('image',image)
    #cv2.waitKey(0)
    image = np.float64(image)
    pad = cv2.BORDER_REFLECT101
    conv_image = np.float64(conv_image)
    conv_image = cv2.filter2D(image,-1,kernel,borderType=pad)
    #conv_image = cv2.sepFilter2D(image, -1, kernel, kernel, borderType = cv2.BORDER_REFLECT101)             
    #conv_image = scipy.signal.convolve2d(image,kernel, boundary ='symm', mode='same')
    conv_image = np.float64(conv_image)
    [m,n] = conv_image.shape
    #print(image.shape)
    red_image = np.zeros(shape=(int(np.ceil(float(m)/2)),int(np.ceil(float(n)/2))))
    red_image = np.float64(red_image)
    for i in range(0,m,2):
       for j in range(0,n,2):
           red_image[int(np.ceil(float(i)/2)),int(np.ceil(float(j)/2))] = conv_image[i,j] 
    #cv2.imshow('red_image',red_image)
    #cv2.waitKey(0)
    #expand_layer(red_image)
    red_image = np.float64(red_image)
    return red_image


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """

    curr = 0
    #image = normalize_and_scale(image)
    #image = image.astype(float)
    image2 = image
    
    output = [image]
    
    #cv2.imshow("pyramid_input",output[curr])
    #cv2.waitKey(0)
    #print(levels)
    for curr in range(0,levels-1):
        reduced_image = reduce_image(image2)
        #reduced_image2 = normalize_and_scale(reduced_image)
        output.append(reduced_image)
        image2 = reduced_image
        #cv2.imshow("gaus",output[curr])
        #cv2.waitKey(0)
    #cv2.imshow('red_image',reduced_image)
    #cv2.waitKey(0)
    # WRITE YOUR CODE HERE.
    #print(len(output))
    return output


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """
    list = img_list
    cols = 0
    row = np.shape(list[0])[0]
    
    for i in range(len(list)):
        cols = cols + np.shape(list[i])[1] 

    combined = np.zeros((row,cols))
    
    start_col = 0
    start_row = 0
    
    for i in range(len(list)):
        end_col = start_col + np.shape(list[i])[1]
        end_row = np.shape(list[i])[0]
        
        combined[start_row:end_row,start_col:end_col] = list[i][:]
        start_col = end_col
        
    
    #print combined
    return normalize_and_scale(combined)


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """
    #cv2.imshow('original image',image)
    #cv2.waitKey(0)   
    
    #print ('input image size =' + str(np.shape(image)))
    [m,n] = np.shape(image)
    a = 0.4
    #kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
    kernel = np.array([1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0])
    kernel = np.outer(kernel,kernel)
    #kernel=generatingKernel(0.4)
    upimage = np.zeros((2*m,2*n))
    upimage = upimage.astype(np.float64)
    for i in range(0,2*m,1):
        for j in range(0,2*n,1):
            if ((i%2)==0) and ((j%2)==0):
                upimage[i,j] = image[i/2,j/2]
    
    #conv_image2 = np.zeros((2*m,2*n))            
    #conv_image2 = conv_image2.astype(float)
    upimage = upimage.astype(np.float64)
    pad = cv2.BORDER_REFLECT101
    
    conv_image2 = cv2.filter2D(upimage,-1,kernel,borderType=pad)     
    #conv_image2 = cv2.sepFilter2D(upimage, -1, kernel, kernel, borderType = cv2.BORDER_REFLECT101)                 
    #conv_image2 = conv_image2.astype(np.float64)
    #cv2.imshow('upimage',conv_image2)
    #cv2.waitKey(0)   
    #print ('output image size =' + str(np.shape(conv_image2)))
    return conv_image2 * 4


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """

    levels = len(g_pyr)
    lappyr = []
    for curr in range(1,levels):
        img = g_pyr[curr]
        [m,n] = g_pyr[curr-1].shape
        next_layer = expand_image(img)
        [p,q] = next_layer.shape
        #print(next_layer.shape)
        if (p>m) or (q>n):
            next_layer = next_layer[0:m,0:n]
        lappyr.append(g_pyr[curr-1]-next_layer)
        #cv2.imshow("level",lappyr[curr-1])
        #cv2.waitKey(0)
    lappyr.append(g_pyr[levels-1])
    #cv2.imshow("level",lappyr[levels-1])
    #cv2.waitKey(0)
    #print('lappyr size')
    #print(len(lappyr))
    # WRITE YOUR CODE HERE.
    #raise NotImplementedError
    return lappyr


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """
    
    M, N = image.shape
    X, Y = np.meshgrid(xrange(N), xrange(M))
    x = X * 1.0 + U * 1.0
    y = Y * 1.0 + V * 1.0
    x_32 = x.astype('float32')
    y_32 = y.astype('float32')
    C = cv2.remap(image, x_32, y_32, interpolation, borderMode=border_mode)
    return C


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """
    
    
    
    #img_a = cv2.blur(img_a,(2,2))
    #img_b = cv2.blur(img_b,(2,2))
    gp_imga = gaussian_pyramid(img_a, levels)
    gp_imgb = gaussian_pyramid(img_b, levels)
    red_img = reduce_image(gp_imga[levels-1])
    m,n = red_img.shape
    uk1 = np.zeros((m,n))
    vk1 = np.zeros((m,n))
    curr_imga = gp_imga[levels-1]
    
    for i in range(levels-1,-1,-1):
        uk1 = expand_image(uk1)
        vk1 = expand_image(vk1)
        uk1 = uk1 * 2
        vk1 = vk1 * 2
        uk1 = uk1[0:gp_imgb[i].shape[0],0:gp_imgb[i].shape[1]]
        vk1 = vk1[0:gp_imgb[i].shape[0],0:gp_imgb[i].shape[1]]
        warp_b = warp(gp_imgb[i], uk1, vk1, interpolation, border_mode)
        (u_corr, v_corr) = optic_flow_lk(gp_imga[i], warp_b, k_size, k_type, sigma=1)
        uk1 = uk1 + u_corr
        vk1 = vk1 + v_corr
    
    
# =============================================================================
#     for i in range(levels-1,0,-1):
#         print 'iteration =' + str(i)
#         curr_imgb = gp_imgb[i]
#         (uk, vk) = optic_flow_lk(curr_imga, curr_imgb, k_size, k_type, sigma=1)
#         #print('uk1 shape =' + str(uk1.shape))
#         #print('uk shape =' + str(uk.shape))
#         uk1 = uk + uk1
#         vk1 = vk + vk1
#         uk1 = expand_image(uk1) * 2
#         vk1 = expand_image(vk1) * 2
#         warped_imga = warp(gp_imgb[i-1], uk1, vk1, interpolation, border_mode)
#         curr_imga = warped_imga
# =============================================================================
#    uk1 = cv2.blur(uk1,(5,5))
#    vk1 = cv2.blur(vk1,(5,5))
    return (uk1,vk1)

def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    
    #print 'in video frame generator'
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    
    video.release()
    
    yield None