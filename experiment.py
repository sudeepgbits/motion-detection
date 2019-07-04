"""Problem Set 4: Motion Detection"""

import cv2
import os
import numpy as np

import ps4

# I/O directories
input_dir = "input_images"
VID_DIR = "input_videos"
output_dir = "output"


# Utility code
def quiver(u, v, scale, stride, color=(0, 255, 0)):

    img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)

    for y in xrange(0, v.shape[0], stride):

        for x in xrange(0, u.shape[1], stride):

            cv2.line(img_out, (x, y), (x + int(u[y, x] * scale),
                                       y + int(v[y, x] * scale)), color, 1)
            cv2.circle(img_out, (x + int(u[y, x] * scale),
                                 y + int(v[y, x] * scale)), 1, color, 1)
    return img_out


# Functions you need to complete:

def scale_u_and_v(u, v, level, pyr):
    """Scales up U and V arrays to match the image dimensions assigned 
    to the first pyramid level: pyr[0].

    You will use this method in part 3. In this section you are asked 
    to select a level in the gaussian pyramid which contains images 
    that are smaller than the one located in pyr[0]. This function 
    should take the U and V arrays computed from this lower level and 
    expand them to match a the size of pyr[0].

    This function consists of a sequence of ps4.expand_image operations 
    based on the pyramid level used to obtain both U and V. Multiply 
    the result of expand_image by 2 to scale the vector values. After 
    each expand_image operation you should adjust the resulting arrays 
    to match the current level shape 
    i.e. U.shape == pyr[current_level].shape and 
    V.shape == pyr[current_level].shape. In case they don't, adjust
    the U and V arrays by removing the extra rows and columns.

    Hint: create a for loop from level-1 to 0 inclusive.

    Both resulting arrays' shapes should match pyr[0].shape.

    Args:
        u: U array obtained from ps4.optic_flow_lk
        v: V array obtained from ps4.optic_flow_lk
        level: level value used in the gaussian pyramid to obtain U 
               and V (see part_3)
        pyr: gaussian pyramid used to verify the shapes of U and V at 
             each iteration until the level 0 has been met.

    Returns:
        tuple: two-element tuple containing:
            u (numpy.array): scaled U array of shape equal to 
                             pyr[0].shape
            v (numpy.array): scaled V array of shape equal to 
                             pyr[0].shape
    """

    # TODO: Your code here
    
    for i in range(level-1,-1,-1):
        curr_img = pyr[i]
        u = expand_image(u) * 2
        v = expand_image(v) * 2
        u = u[0:curr_img.shape[0],0:curr_img.shape[1]]
        v = v[0:curr_img.shape[0],0:curr_img.shape[1]]
        
        
    return (u,v)


def part_1a():

    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r2 = cv2.imread(os.path.join(input_dir, 'TestSeq', 
                                       'ShiftR2.png'), 0) / 255.
    shift_r5_u5 = cv2.imread(os.path.join(input_dir, 'TestSeq', 
                                          'ShiftR5U5.png'), 0) / 255.

    # Optional: smooth the images if LK doesn't work well on raw images
    k_size = 15  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r2, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-1.png"), u_v)

    # Now let's try with ShiftR5U5. You may want to try smoothing the
    # input images first.

    k_size = 65  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 15  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r5_u5, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-2.png"), u_v)


def part_1b():
    """Performs the same operations applied in part_1a using the images
    ShiftR10, ShiftR20 and ShiftR40.

    You will compare the base image Shift0.png with the remaining
    images located in the directory TestSeq:
    - ShiftR10.png
    - ShiftR20.png
    - ShiftR40.png

    Make sure you explore different parameters and/or pre-process the
    input images to improve your results.

    In this part you should save the following images:
    - ps4-1-b-1.png
    - ps4-1-b-2.png
    - ps4-1-b-3.png

    Returns:
        None
    """
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.
    
                                        
    k_size = 35  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r10, k_size, k_type, sigma)
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-1.png"), u_v)
    
    k_size = 35  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r20, k_size, k_type, sigma)
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-2.png"), u_v)
    
    k_size = 35  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r40, k_size, k_type, sigma)
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-3.png"), u_v)
    
                                        
    


def part_2():

    yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1',
                                         'yos_img_01.jpg'), 0) / 255.

    # 2a
    levels = 4
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_01_g_pyr_img = ps4.create_combined_img(yos_img_01_g_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-a-1.png"),
                yos_img_01_g_pyr_img)

    # 2b
    yos_img_01_l_pyr = ps4.laplacian_pyramid(yos_img_01_g_pyr)

    yos_img_01_l_pyr_img = ps4.create_combined_img(yos_img_01_l_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-b-1.png"),
                yos_img_01_l_pyr_img)


def part_3a_1():
    yos_img_01 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.

    levels = 1  # Define the number of pyramid levels
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)

    level_id = 0  # TODO: Select the level number (or id) you wish to use
    k_size = 45  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 15  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_01_g_pyr[level_id],
                             yos_img_02_g_pyr[level_id],
                             k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_02_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_02_warped = ps4.warp(yos_img_02, u, v, interpolation, border_mode)

    diff_yos_img_01_02 = yos_img_01 - yos_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-1.png"),
                ps4.normalize_and_scale(diff_yos_img_01_02))


def part_3a_2():
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.
    yos_img_03 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_03.jpg'), 0) / 255.

    levels = 1  # Define the number of pyramid levels
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)
    yos_img_03_g_pyr = ps4.gaussian_pyramid(yos_img_03, levels)

    level_id = 0  # TODO: Select the level number (or id) you wish to use
    k_size = 45  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 15  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_02_g_pyr[level_id],
                             yos_img_03_g_pyr[level_id],
                             k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_03_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_03_warped = ps4.warp(yos_img_03, u, v, interpolation, border_mode)

    diff_yos_img = yos_img_02 - yos_img_03_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-2.png"),
                ps4.normalize_and_scale(diff_yos_img))


def part_4a():
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.

    levels = 6  # TODO: Define the number of levels
    k_size = 20  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 65  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u10, v10 = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u10, v10, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-1.png"), u_v)

    # You may want to try different parameters for the remaining function
    # calls.
    u20, v20 = ps4.hierarchical_lk(shift_0, shift_r20, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u20, v20, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-2.png"), u_v)

    u40, v40 = ps4.hierarchical_lk(shift_0, shift_r40, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)
    u_v = quiver(u40, v40, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-3.png"), u_v)


def part_4b():
    urban_img_01 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban01.png'), 0) / 255.
    urban_img_02 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban02.png'), 0) / 255.

    levels = 3  # TODO: Define the number of levels
    k_size = 100  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 65  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u, v = ps4.hierarchical_lk(urban_img_01, urban_img_02, levels, k_size,
                               k_type, sigma, interpolation, border_mode)

    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-1.png"), u_v)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    urban_img_02_warped = ps4.warp(urban_img_02, u, v, interpolation,
                                   border_mode)

    diff_img = urban_img_01 - urban_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-2.png"),
                ps4.normalize_and_scale(diff_img))


def part_5a():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    
    I0 = cv2.imread(
        os.path.join(input_dir, 'TestSeq', 'Shift0.png'), 0) / 255.
    I1 = cv2.imread(
        os.path.join(input_dir, 'TestSeq', 'ShiftR10.png'), 0) / 255.
            
            
#    cv2.imshow('I0 image', I0)
#    cv2.imshow('I1 image', I1)
#    cv2.waitKey(0)
    levels = 3  # TODO: Define the number of levels
    k_size = 65  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 65  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    
    
#    final_output = np.zeros((3*I0.shape[0],3*I0.shape[1]))
#    start_col = 0
#    start_row = 0
    u, v = ps4.hierarchical_lk(I0, I1, levels, k_size, k_type, sigma, interpolation,border_mode)
    t = 0.0
    filename = 'intermediate ' + str(t) + '.png'
    cv2.imwrite(os.path.join(output_dir, filename), ps4.normalize_and_scale(I0))
    t = 0.2
    final_output1 = I0
    while t <= 1.0:
        
        It = (1-t)*I0 + t*ps4.warp(I1, u, v, interpolation, border_mode)
        It = ps4.warp(I0, -t*u, -t*v, interpolation, border_mode)
#        cv2.imshow('intermediate', It)
#        cv2.waitKey(0)
        filename = 'intermediate ' + str(t) + '.png'
        cv2.imwrite(os.path.join(output_dir, filename), ps4.normalize_and_scale(It))
        #print t
        if t<=0.4:
            final_output1 = np.hstack((final_output1,It))
        elif t > 0.6 and t < 0.7:
            #print 't == 0.6'
            final_output2 = It
        elif t > 0.7:
            #print 't > 0.6'
            final_output2 = np.hstack((final_output2,It))
        
        t = t + 0.2
        
    
    final_output = np.vstack((final_output1,final_output2))
    cv2.imwrite(os.path.join(output_dir, "ps4-5-1-a-1.png"), ps4.normalize_and_scale(final_output))
#    cv2.imshow('final ooutput', final_output)
#    cv2.waitKey(0)


def part_5b():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    output_dir = "output"
    I0 = cv2.imread(
        os.path.join(input_dir, 'MiniCooper', 'mc01.png'), 0) / 255.
    I1 = cv2.imread(
        os.path.join(input_dir, 'MiniCooper', 'mc02.png'), 0) / 255.
    I2 = cv2.imread(
        os.path.join(input_dir, 'MiniCooper', 'mc03.png'), 0) / 255.
    
            
#    I0 = cv2.blur(I0,(5,5))
#    I1 = cv2.blur(I1,(5,5))
#    I2 = cv2.blur(I2,(5,5))
    
    image_list = []
    image_list.append(I0)
    image_list.append(I1)
    image_list.append(I2)            
#    cv2.imshow('I0 image', I0)
#    cv2.imshow('I1 image', I1)
#    cv2.waitKey(0)
    levels = 6  # TODO: Define the number of levels
    k_size = 40 # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 65  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    
    
#    final_output = np.zeros((3*I0.shape[0],3*I0.shape[1]))
#    start_col = 0
#    start_row = 0
    
    for i in range(len(image_list)-1):
        I0 = image_list[i]
        I1 = image_list[i+1]
    
        u, v = ps4.hierarchical_lk(I0, I1, levels, k_size, k_type, sigma, interpolation,border_mode)
        t = 0.0
        filename = 'intermediate ' +str(i) + '.' + str(t) + '.png'
        cv2.imwrite(os.path.join(output_dir, filename), ps4.normalize_and_scale(I0))
        t = 0.2
        final_output1 = I0
        while t <= 1.0:
            
            It = (1-t)*I0 + t*ps4.warp(I1, u, v, interpolation, border_mode)
            #It = ps4.warp(I0, -t*u, -t*v, interpolation, border_mode)
            #It = 0.4*It1 + 0.6* It2
    #        cv2.imshow('intermediate', It)
    #        cv2.waitKey(0)
            filename = 'intermediate ' +str(i) + '.' + str(t) + '.png'
            cv2.imwrite(os.path.join(output_dir, filename), ps4.normalize_and_scale(It))
            #print t
            if t<=0.4:
                final_output1 = np.hstack((final_output1,It))
            elif t > 0.6 and t < 0.7:
                #print 't == 0.6'
                final_output2 = It
            elif t > 0.7:
                #print 't > 0.6'
                final_output2 = np.hstack((final_output2,It))
            
            t = t + 0.2
            
        
        output_file  = 'ps4-5-1-b-' + str(i+1)+ '.png'
        final_output = np.vstack((final_output1,final_output2))
        cv2.imwrite(os.path.join(output_dir, output_file), ps4.normalize_and_scale(final_output))
    
    
    
    
    


def part_6():
    """Challenge Problem

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    
  
    
    video_file = "ps4-my-video.mp4"  # Place your video in the input_video directory
    frame_ids = [55, 60]
    fps = 10
    
    # Todo: Complete this part on your own.        
    

    video= os.path.join(VID_DIR, video_file)
    video_gen = ps4.video_frame_generator(video)

    video_image = video_gen.next()
    
    #print video_image
    #cv2.imshow('frame', video_image)
    #cv2.waitKey(0)
    h, w, d = video_image.shape

    output_name = "ps4-6-a-2"
    out_path = "output/ar_{}-{}".format(output_name[4:], video_file)
    video_final = mp4_video_writer(out_path, (w, h), fps)
    
    levels = 3  # TODO: Define the number of levels
    k_size = 140  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 65  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    
    counter_init = 1
    output_counter = counter_init
    output_name = "ps4-6-a"
    frame_num = 1
    image_counter = 1
    for frame_num in range(200):
        video_image = video_gen.next()
        frame_num = frame_num + 1
    
    frame_num = 1
    while video_image is not None and frame_num<250 :
        print ("Processing video frame" + str(frame_num)) 
        
        if frame_num < 2:
            prev_frame = video_image
            prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            
        else:
            curr_frame = video_image
            curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            u, v = ps4.hierarchical_lk(prev_frame_gray, curr_frame_gray, levels, k_size, k_type, sigma, interpolation,border_mode)
            u_v = quiver2(prev_frame, u, v, scale=3, stride=10)
            
            #cv2.imshow('output', u_v)
            
            frame_id = frame_ids[(output_counter - 1) % 3]
    
            if frame_num == 150 or frame_num == 200:
                out_str = output_name + "-{}.png".format(output_counter)
                save_image(out_str, curr_frame)
                image_counter += 1
    
            video_final.write(u_v)
    
            video_image = video_gen.next()
            prev_frame = curr_frame
            prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frame_num += 1

    video_final.release()


def save_image(filename, image):
    """Convenient wrapper for writing images to the output directory."""
    cv2.imwrite(os.path.join(output_dir, filename), image)


def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    fourcc = cv2.cv.CV_FOURCC(*'MP4V')
    filename = filename.replace('.mp4', '.avi')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)    

def quiver2(frame, u, v, scale, stride, color=(0, 255, 0)):

    img_out = frame

    for y in xrange(0, v.shape[0], stride):

        for x in xrange(0, u.shape[1], stride):

            cv2.line(img_out, (x, y), (x + int(u[y, x] * scale),
                                       y + int(v[y, x] * scale)), color, 1)
            cv2.circle(img_out, (x + int(u[y, x] * scale),
                                 y + int(v[y, x] * scale)), 1, color, 1)
    return img_out




if __name__ == "__main__":
    part_1a()
    part_1b()
    part_2()
    part_3a_1()
    part_3a_2()
    part_4a()
    part_4b()
    part_5a()
    part_5b()
    part_6()
