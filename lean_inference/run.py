from lean_inference.infere import Infere_model_and_show_result
# from lean_inference.infere_and_save_as_jpgs import Infere_model_and_save_result
if __name__ == "__main__":
    model_name = 'my_conv_net'
    # model_name = 'hybrid'
    # model_name = 'inception_v3'

    images_parent_dir = 'D:\\phantomAI\\data\\collected_data'

    # Infere_model_and_save_result(model_name, images_parent_dir, target_dir)
    Infere_model_and_show_result(model_name, images_parent_dir)