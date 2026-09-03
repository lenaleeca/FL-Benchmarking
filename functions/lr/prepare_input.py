from functions.mlp.prepare_input import (
    find_test_partition_file_in_site_result_dir,
    get_test_partition_csv,
    load_input_data,
    prepare_mlp_input,
)

# LR use same tabular input structure and preprocessing as MLP
def prepare_lr_input(csv_path, *, expected_n_features=None, label_col=None, scale=False, drop_cols=True):
    return prepare_mlp_input(
        csv_path,
        expected_n_features=expected_n_features,
        label_col=label_col,
        scale=scale,
        drop_cols=drop_cols,
    )
