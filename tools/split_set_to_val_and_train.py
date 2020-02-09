import shutil
import os
from tools.random_tools import get_rand_out_of_list_item


def create_dirs_for_train_and_vals_sets(dir_name):

    train_dir_name = os.path.join(dir_name, 'train')
    val_dir_name = os.path.join(dir_name, 'val')
    if not os.path.isdir(train_dir_name):
        os.mkdir(train_dir_name)
    else:
        print("ther is allready train directory here")
        return False

    os.mkdir(val_dir_name)

    new_md_val_dir_name = os.path.join(val_dir_name, 'meta_data')
    new_fv_val_dir_name = os.path.join(val_dir_name, 'front_view_image')
    new_tv_val_dir_name = os.path.join(val_dir_name, 'top_view_image')
    new_tvv_val_dir_name = os.path.join(val_dir_name, 'top_view_image_with_vehicles')

    os.mkdir(new_md_val_dir_name)
    os.mkdir(new_fv_val_dir_name)
    os.mkdir(new_tv_val_dir_name)
    os.mkdir(new_tvv_val_dir_name)

    new_md_train_dir_name = os.path.join(train_dir_name, 'meta_data')
    new_fv_train_dir_name = os.path.join(train_dir_name, 'front_view_image')
    new_tv_train_dir_name = os.path.join(train_dir_name, 'top_view_image')
    new_tvv_train_dir_name = os.path.join(train_dir_name, 'top_view_image_with_vehicles')

    os.mkdir(new_md_train_dir_name)
    os.mkdir(new_fv_train_dir_name)
    os.mkdir(new_tv_train_dir_name)
    os.mkdir(new_tvv_train_dir_name)

    return True

def split_to_train_and_val_sets(dir_name):

    create_dirs_for_train_and_vals_sets(dir_name)
    # if not is_new:
    #     print(" THERE IS SOMETHING HERE>>>")
    #     return

    md_dir = os.path.join(dir_name, 'meta_data')
    fv_dir = os.path.join(dir_name, 'front_view_image')
    tv_dir = os.path.join(dir_name, 'top_view_image')
    tvv_dir = os.path.join(dir_name, 'top_view_image_with_vehicles')
    meta_data_files_list = os.listdir(md_dir)

    for i, md_filename in enumerate(meta_data_files_list):
        if 'meta_data' not in md_filename:
            continue
        md_png_filename = md_filename.replace('json', 'png')
        md_fn = os.path.join(md_dir, md_filename)
        fv_dbg_fn = os.path.join(fv_dir, md_png_filename.replace('meta_data', 'front_view_image'))
        fv_seg_fn = os.path.join(fv_dir, md_png_filename.replace('meta_data', 'seg_front_view_image'))
        fv_seg_crop_fn = os.path.join(fv_dir, md_png_filename.replace('meta_data', 'seg_crop_front_view_image'))
        tv_fn = os.path.join(tv_dir, md_png_filename.replace('meta_data', 'top_view_image'))
        tvv_fn = os.path.join(tvv_dir, md_png_filename.replace('meta_data', 'top_view_image_vcls'))
        
        dest_dir_type = get_rand_out_of_list_item(objects_list=['train', 'val'], objects_weights=[0.8, 0.2])
        dest_dir_name = os.path.join(dir_name, dest_dir_type)
        new_md_dir = os.path.join(dest_dir_name, "meta_data")
        new_fv_dir = os.path.join(dest_dir_name, "front_view_image")
        new_tv_dir = os.path.join(dest_dir_name, "top_view_image")
        new_tvv_dir = os.path.join(dest_dir_name, "top_view_image_with_vehicles")

        shutil.copy(fv_dbg_fn, new_fv_dir)
        shutil.copy(fv_seg_fn, new_fv_dir)
        shutil.copy(fv_seg_crop_fn, new_fv_dir)
        shutil.copy(md_fn, new_md_dir)
        shutil.copy(tv_fn, new_tv_dir)
        shutil.copy(tvv_fn, new_tvv_dir)

        if i % 100 == 0:
            print(i, " 100's of ", len(meta_data_files_list))


if __name__ == '__main__':
    dir_name_ = 'D:\\phantomAI\\data\\synthesized_data\\testing\\2019_12_11__20_32_run_'
    split_to_train_and_val_sets(dir_name_)