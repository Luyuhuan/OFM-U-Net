#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import multiprocessing
import shutil
from time import sleep
from typing import Union, Tuple

import nnunetv2
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import get_identifiers_from_splitted_dataset_folder, \
    create_lists_from_splitted_dataset_folder, get_filenames_of_train_images_and_targets
from tqdm import tqdm
lzz_flag = 1
from batchgenerators.utilities.file_and_folder_operations import load_pickle
import os
import concurrent.futures
import numpy as np
import cv2

def compute_flow_batch(frames_np):
    # 初始化光流数组
    flows = np.zeros((frames_np.shape[0], frames_np.shape[2], frames_np.shape[3], 2), dtype=np.float32)
    flowsbefore = np.zeros((frames_np.shape[0], frames_np.shape[2], frames_np.shape[3], 2), dtype=np.float32)
    # 对批次中的每对连续帧计算光流
    for idx in range(frames_np.shape[0] - 1):
        frame1 = np.moveaxis(frames_np[idx], 0, -1)  # C, H, W -> H, W, C
        frame2 = np.moveaxis(frames_np[idx + 1], 0, -1)  # C, H, W -> H, W, C

        # 转换为灰度图像
        frame1_gray = cv2.cvtColor(frame1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        frame2_gray = cv2.cvtColor(frame2.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        flows[idx] = flow
        # flowsbefore[idx + 1] = flow
    return flows, flowsbefore
def letterbox(im, new_shape=(192,192), color=(0,0,0), auto=False, scaleFill=False, scaleup=True, stride=1):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    im = np.array(im)
    im = im.transpose(1,2,0)
    # im shape  (H, W, C)
    # filepath = r"/root/lzz2/cardiac_cycle/mamba/U-Mamba/data/showorg.png"
    # cv2.imwrite(filepath, im)
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # filepath = r"/root/lzz2/cardiac_cycle/mamba/U-Mamba/data/show224.png"
    # cv2.imwrite(filepath, im)
    im = im.transpose(2,0,1)
    # im shape (C,H,W)
    return im

# Assuming 'data' is the list of arrays and 'letterbox' is the letterbox function

def process_item(nowdata):
    return letterbox(nowdata)


class DefaultPreprocessor(object):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        """
        Everything we need is in the plans. Those are given when run() is called
        """
    def run_case_npy(self, videodata, seg: Union[np.ndarray, None], properties: dict,
                     plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                     dataset_json: Union[dict, str]):
        # let's not mess up the inputs!
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], "Shape mismatch between image and segmentation. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
            seg = np.copy(seg)

        has_seg = seg is not None
        # lzz append
        global lzz_flag
        data_list = [[],[]]
        seg_list = []
        if lzz_flag == 1:
            # needimg = 48
            needimg = 40
            needshape = (3, 192,192)
            #  ************************************************* 处理视频  *************************************************
            # 记录选取帧的起始  方便取json对应的值
            annostart = 0
            annoend = 0
            #  读取视频
            cap = cv2.VideoCapture(videodata)
            if not cap.isOpened():
                print(f"无法打开视频文件：{videodata}")
                return None
            # 获取视频的总帧数
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if needimg > frame_count -1:
                frames = []
                startindex = 1
                split_start = startindex
                # 跳到起始帧
                annostart = split_start
                annoend = frame_count
                cap.set(cv2.CAP_PROP_POS_FRAMES, split_start)
                for i in range(split_start, frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                frames = np.array(frames)
                data_list[0].append([annostart,annoend])
                data_list[1].append(frames)
            else:
                startindex = 1
                endindex = frame_count - needimg
                for split_start in range(startindex, endindex, 5):
                    frames = []
                    annostart = split_start
                    annoend = split_start + needimg
                    # 跳到起始帧
                    cap.set(cv2.CAP_PROP_POS_FRAMES, split_start)
                    for i in range(split_start, split_start + needimg):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(frame)
                    frames = np.array(frames)
                    data_list[0].append([annostart,annoend])
                    data_list[1].append(frames)
                # 单独把最后面保留
                # print("test change frame")
                split_start = frame_count - needimg
                frames = []
                annostart = split_start
                annoend = split_start + needimg
                cap.set(cv2.CAP_PROP_POS_FRAMES, split_start)
                for i in range(split_start, split_start + needimg):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                frames = np.array(frames)
                data_list[0].append([annostart,annoend])
                data_list[1].append(frames)
            for dataindex,data in enumerate(data_list[1]):
                assert np.min(data)>= 0, "RGB images are uint 8, for whatever reason I found pixel values smaller than 0. " \
                                    "Your images do not seem to be RGB images"
                assert np.max(data) <= 255, "RGB images are uint 8, for whatever reason I found pixel values greater than 255" \
                                        ". Your images do not seem to be RGB images"
                data = data / 255.
                data = np.transpose(data, (0,3,1,2))
                num_threads = 4  # Adjust the number of threads as per your requirements
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
                # Process the items in parallel using the ThreadPoolExecutor
                newdata = []
                with executor:
                    futures = [executor.submit(process_item, nowdata) for nowdata in data]
                    for future in concurrent.futures.as_completed(futures):
                        nowdata = future.result()
                        newdata.append(nowdata)
                # Convert the processed data list to a NumPy array
                needtap = needimg - len(newdata)
                if needtap > 0:
                    for i in range(needtap):
                        zeros_array = np.zeros(needshape)
                        newdata.append(zeros_array)
                newdata = np.transpose(np.stack(newdata, axis=0), (1,0,2,3))                
                #  ************************************************* 加上光流  *************************************************
                flowflag = 0
                if flowflag == 1:
                    video_frames_tensor = np.transpose(newdata, (1,0,2,3)) * 255
                    flows,flowsbefore = compute_flow_batch(video_frames_tensor)
                    newdata = np.concatenate((np.transpose(flows, (3,0,1,2)), newdata), axis=0)
                    # newdata = np.concatenate((np.transpose(flowsbefore, (3,0,1,2)), np.transpose(flows, (3,0,1,2)), newdata), axis=0)
                #  ************************************************* 加上光流  *************************************************
                data_list[1][dataindex] = newdata
            #  ************************************************* 处理regression  *************************************************
            if needtap > 0 and seg is not None:
                seg = seg.tolist()
                needarray = np.array([2.0])
                for i in range(needtap):
                    seg.append(needarray)
                seg = np.array(seg)
            return data_list, seg
    
    def run_case_npy_org(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict,
                     plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                     dataset_json: Union[dict, str]):
        # let's not mess up the inputs!
        data = np.copy(data)
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], "Shape mismatch between image and segmentation. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
            seg = np.copy(seg)

        has_seg = seg is not None
        # lzz append
        global lzz_flag
        if lzz_flag == 1:
            # needimg = 48
            needimg = 40
            needshape = (3, 192,192)
            #  ************************************************* 处理image  *************************************************
            #  ************************************************* 处理image  *************************************************
            assert np.min(data)>= 0, "RGB images are uint 8, for whatever reason I found pixel values smaller than 0. " \
                                "Your images do not seem to be RGB images"
            assert np.max(data) <= 255, "RGB images are uint 8, for whatever reason I found pixel values greater than 255" \
                                    ". Your images do not seem to be RGB images"
            data = data / 255.
            data = np.transpose(data, (1,0,3,2))
            # print(newdata)
            # newdata = []
            # for nowdata in data:
            #     nowdata = letterbox(nowdata)
            #     newdata.append(nowdata)
            # newdata = np.stack(newdata, axis=0)
            # Create a ThreadPoolExecutor with the desired number of threads
            num_threads = 4  # Adjust the number of threads as per your requirements
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
            # Process the items in parallel using the ThreadPoolExecutor
            newdata = []
            with executor:
                futures = [executor.submit(process_item, nowdata) for nowdata in data]
                for future in concurrent.futures.as_completed(futures):
                    nowdata = future.result()
                    newdata.append(nowdata)
            # Convert the processed data list to a NumPy array
            needtap = needimg - len(newdata)
            if needtap > 0:
                for i in range(needtap):
                    zeros_array = np.zeros(needshape)
                    newdata.append(zeros_array)
            newdata = np.transpose(np.stack(newdata, axis=0), (1,0,2,3))
            data = newdata
            #  ************************************************* 处理regression  *************************************************
            if needtap > 0 and seg is not None:
                seg = seg.tolist()
                needarray = np.array([2.0])
                for i in range(needtap):
                    seg.append(needarray)
                seg = np.array(seg)
            return data, seg
        # apply transpose_forward, this also needs to be applied to the spacing!
        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if seg is not None:
            seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, bbox = crop_to_nonzero(data, seg)
        properties['bbox_used_for_cropping'] = bbox
        # print(data.shape, seg.shape)
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # resample
        target_spacing = configuration_manager.spacing  # this should already be transposed

        if len(target_spacing) < len(data.shape[1:]):
            # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 2d configuration we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        data = self._normalize(data, seg, configuration_manager,
                               plans_manager.foreground_intensity_properties_per_channel)

        # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
        old_shape = data.shape[1:]
        data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
        seg = configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing)
        if self.verbose:
            print(f'old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, '
                  f'new_spacing: {target_spacing}, fn_data: {configuration_manager.resampling_fn_data}')

        # if we have a segmentation, sample foreground locations for oversampling and add those to properties
        if has_seg:
            # reinstantiating LabelManager for each case is not ideal. We could replace the dataset_json argument
            # with a LabelManager Instance in this function because that's all its used for. Dunno what's better.
            # LabelManager is pretty light computation-wise.
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect_for_this = label_manager.foreground_regions if label_manager.has_regions \
                else label_manager.foreground_labels

            # when using the ignore label we want to sample only from annotated regions. Therefore we also need to
            # collect samples uniformly from all classes (incl background)
            if label_manager.has_ignore_label:
                collect_for_this.append(label_manager.all_labels)

            # no need to filter background in regions because it is already filtered in handle_labels
            # print(all_labels, regions)
            properties['class_locations'] = self._sample_foreground_locations(seg, collect_for_this,
                                                                                   verbose=self.verbose)
            seg = self.modify_seg_fn(seg, plans_manager, dataset_json, configuration_manager)
        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)
        return data, seg

    def run_case(self, image_files: List[str], seg_file: Union[str, None], plans_manager: PlansManager,
                 configuration_manager: ConfigurationManager,
                 dataset_json: Union[dict, str]):
        """
        seg file can be none (test cases)

        order of operations is: transpose -> crop -> resample
        so when we export we need to run the following order: resample -> crop -> transpose (we could also run
        transpose at a different place, but reverting the order of operations done during preprocessing seems cleaner)
        """
        global lzz_flag
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)

        rw = plans_manager.image_reader_writer_class()

        # load image(s)
        if lzz_flag == 1:
            # data = np.load(image_files[0], 'r')
            # data_properties = load_pickle(os.path.join(os.path.dirname(image_files[0]),"all.pkl"))
            data = image_files[0]
            pklfile = r"/root/lzz2/cardiac_cycle/mamba/try_umamba/U-Mamba/data/nnUNet_preprocessed/Dataset701_AbdomenCT/Apical_4C/all.pkl" 
            # pklfile = r"/root/lzz2/cardiac_cycle/mamba/try_umamba/U-Mamba/data/nnUNet_preprocessed/Dataset701_AbdomenCT/Parasternal_4C/all.pkl" 
            # pklfile = r"/root/lzz2/cardiac_cycle/mamba/try_umamba/U-Mamba/data/nnUNet_preprocessed/Dataset701_AbdomenCT/Basal_4C/all.pkl" 
            data_properties = load_pickle(pklfile)
        else:
            data, data_properties = rw.read_images(image_files)

        # if possible, load seg
        if seg_file is not None:
            seg, _ = rw.read_seg(seg_file)
        else:
            seg = None

        data, seg = self.run_case_npy(data, seg, data_properties, plans_manager, configuration_manager,
                                      dataset_json)
        return data, seg, data_properties

    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                      dataset_json: Union[dict, str]):
        data, seg, properties = self.run_case(image_files, seg_file, plans_manager, configuration_manager, dataset_json)
        # print('dtypes', data.dtype, seg.dtype)
        np.savez_compressed(output_filename_truncated + '.npz', data=data, seg=seg)
        write_pickle(properties, output_filename_truncated + '.pkl')

    @staticmethod
    def _sample_foreground_locations(seg: np.ndarray, classes_or_regions: Union[List[int], List[Tuple[int, ...]]],
                                     seed: int = 1234, verbose: bool = False):
        num_samples = 10000
        min_percent_coverage = 0.01  # at least 1% of the class voxels need to be selected, otherwise it may be too
        # sparse
        rndst = np.random.RandomState(seed)
        class_locs = {}
        for c in classes_or_regions:
            k = c if not isinstance(c, list) else tuple(c)
            if isinstance(c, (tuple, list)):
                mask = seg == c[0]
                for cc in c[1:]:
                    mask = mask | (seg == cc)
                all_locs = np.argwhere(mask)
            else:
                all_locs = np.argwhere(seg == c)
            if len(all_locs) == 0:
                class_locs[k] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

            selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
            class_locs[k] = selected
            if verbose:
                print(c, target_num_samples)
        return class_locs

    def _normalize(self, data: np.ndarray, seg: np.ndarray, configuration_manager: ConfigurationManager,
                   foreground_intensity_properties_per_channel: dict) -> np.ndarray:
        for c in range(data.shape[0]):
            scheme = configuration_manager.normalization_schemes[c]
            normalizer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "preprocessing", "normalization"),
                                                           scheme,
                                                           'nnunetv2.preprocessing.normalization')
            if normalizer_class is None:
                raise RuntimeError(f'Unable to locate class \'{scheme}\' for normalization')
            normalizer = normalizer_class(use_mask_for_norm=configuration_manager.use_mask_for_norm[c],
                                          intensityproperties=foreground_intensity_properties_per_channel[str(c)])
            data[c] = normalizer.run(data[c], seg[0])
        return data

    def run(self, dataset_name_or_id: Union[int, str], configuration_name: str, plans_identifier: str,
            num_processes: int):
        """
        data identifier = configuration name in plans. EZ.
        """
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)

        assert isdir(join(nnUNet_raw, dataset_name)), "The requested dataset could not be found in nnUNet_raw"

        plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + '.json')
        assert isfile(plans_file), "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment " \
                                   "first." % plans_file
        plans = load_json(plans_file)
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(configuration_name)

        if self.verbose:
            print(f'Preprocessing the following configuration: {configuration_name}')
        if self.verbose:
            print(configuration_manager)

        dataset_json_file = join(nnUNet_preprocessed, dataset_name, 'dataset.json')
        dataset_json = load_json(dataset_json_file)

        output_directory = join(nnUNet_preprocessed, dataset_name, configuration_manager.data_identifier)

        if isdir(output_directory):
            shutil.rmtree(output_directory)

        maybe_mkdir_p(output_directory)

        dataset = get_filenames_of_train_images_and_targets(join(nnUNet_raw, dataset_name), dataset_json)

        # identifiers = [os.path.basename(i[:-len(dataset_json['file_ending'])]) for i in seg_fnames]
        # output_filenames_truncated = [join(output_directory, i) for i in identifiers]

        # multiprocessing magic.
        r = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            for k in dataset.keys():
                r.append(p.starmap_async(self.run_case_save,
                                         ((join(output_directory, k), dataset[k]['images'], dataset[k]['label'],
                                           plans_manager, configuration_manager,
                                           dataset_json),)))
            remaining = list(range(len(dataset)))
            # p is pretty nifti. If we kill workers they just respawn but don't do any work.
            # So we need to store the original pool of workers.
            workers = [j for j in p._pool]
            with tqdm(desc=None, total=len(dataset), disable=self.verbose) as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError('Some background worker is 6 feet under. Yuck. \n'
                                           'OK jokes aside.\n'
                                           'One of your background processes is missing. This could be because of '
                                           'an error (look for an error message) or because it was killed '
                                           'by your OS due to running out of RAM. If you don\'t see '
                                           'an error message, out of RAM is likely the problem. In that case '
                                           'reducing the number of workers might help')
                    done = [i for i in remaining if r[i].ready()]
                    for _ in done:
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)

    def modify_seg_fn(self, seg: np.ndarray, plans_manager: PlansManager, dataset_json: dict,
                      configuration_manager: ConfigurationManager) -> np.ndarray:
        # this function will be called at the end of self.run_case. Can be used to change the segmentation
        # after resampling. Useful for experimenting with sparse annotations: I can introduce sparsity after resampling
        # and don't have to create a new dataset each time I modify my experiments
        return seg


def example_test_case_preprocessing():
    # (paths to files may need adaptations)
    plans_file = '/home/isensee/drives/gpu_data/nnUNet_preprocessed/Dataset219_AMOS2022_postChallenge_task2/nnUNetPlans.json'
    dataset_json_file = '/home/isensee/drives/gpu_data/nnUNet_preprocessed/Dataset219_AMOS2022_postChallenge_task2/dataset.json'
    input_images = ['/home/isensee/drives/e132-rohdaten/nnUNetv2/Dataset219_AMOS2022_postChallenge_task2/imagesTr/amos_0600_0000.nii.gz', ]  # if you only have one channel, you still need a list: ['case000_0000.nii.gz']

    configuration = '3d_fullres'
    pp = DefaultPreprocessor()

    # _ because this position would be the segmentation if seg_file was not None (training case)
    # even if you have the segmentation, don't put the file there! You should always evaluate in the original
    # resolution. What comes out of the preprocessor might have been resampled to some other image resolution (as
    # specified by plans)
    plans_manager = PlansManager(plans_file)
    data, _, properties = pp.run_case(input_images, seg_file=None, plans_manager=plans_manager,
                                      configuration_manager=plans_manager.get_configuration(configuration),
                                      dataset_json=dataset_json_file)

    # voila. Now plug data into your prediction function of choice. We of course recommend nnU-Net's default (TODO)
    return data


if __name__ == '__main__':
    example_test_case_preprocessing()
    # pp = DefaultPreprocessor()
    # pp.run(2, '2d', 'nnUNetPlans', 8)

    ###########################################################################################################
    # how to process a test cases? This is an example:
    # example_test_case_preprocessing()
