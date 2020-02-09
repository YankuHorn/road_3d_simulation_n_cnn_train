import os


def get_best_loss_in_logs():
    parent = 'D:\\phantomAI\\results\\'

    # best_5_val_loss = [100, 100, 100, 100, 100]
    # best_5_train_loss = [100, 100, 100, 100, 100]



    model_names = os.listdir(parent)

    for model_name in model_names:
        best_loss_train = 100
        best_loss_val = 100

        best_loss_train_model_name = None
        best_loss_val_model_name = None
        best_loss_val_idx = -1
        best_loss_train_idx = -1
        if os.path.isdir(os.path.join(parent, model_name)):
            model_dir_name = os.path.join(parent, model_name)
            model_version_names = os.listdir(model_dir_name)

            for model_version in model_version_names:
                model_verion_dir = os.path.join(model_dir_name, model_version)
                for file in os.listdir(model_verion_dir):
                    if file.endswith('log.txt'):
                        ffn = os.path.join(model_verion_dir, file)
                        with open(ffn) as f:
                            for idx, line in enumerate(f):
                                a, b = line.split()
                                val_loss = float(b)
                                train_loss = float(a)
                                if val_loss < best_loss_val:
                                    best_loss_val = val_loss
                                    best_loss_val_model_name = model_version
                                    best_loss_val_idx = idx
                                if train_loss < best_loss_train:
                                    best_loss_train = train_loss
                                    best_loss_train_model_name = model_version
                                    best_loss_train_idx = idx
            print(best_loss_val, "best loss val for", model_name, " is: ", "best_loss_val_model_name", best_loss_val_model_name, best_loss_val_idx)
            # print("best loss train for", model_name, " is: ", best_loss_train, "best_loss_train_model_name", best_loss_train_model_name, best_loss_train_idx)


if __name__ == '__main__':
    get_best_loss_in_logs()