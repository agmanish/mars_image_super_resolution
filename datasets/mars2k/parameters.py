class mars2kParameters:
    def __init__(self, dataset_key):
        if dataset_key not in available_datasets.keys():
            raise ValueError(f"available datasets are: {available_datasets.keys()}")

        dataset_parameters = available_datasets[dataset_key]

        self.train_directory = dataset_parameters["train_directory"]
        self.valid_directory = dataset_parameters["valid_directory"]
        self.scale = dataset_parameters["scale"]


available_datasets = {
    "bicubic_x2": {
        "train_directory": "mars2k_dataset/LRx2/train/",
        "valid_directory": "mars2k_dataset/LRx2/valid/",
        "scale": 2
    },
    
    "bicubic_x3": {
        "train_directory": "mars2k_dataset/LRx3/train/",
        "valid_directory": "mars2k_dataset/LRx3/valid/",
        "scale": 3
    },
    
    "bicubic_x4": {
        "train_directory": "mars2k_dataset/LRx4/train/",
        "valid_directory": "mars2k_dataset/LRx4/valid/",
        "scale": 4
    },
    
    "bicubic_x8": {
        "train_directory": "mars2k_dataset/LRx8/train/",
        "valid_directory": "mars2k_dataset/LRx8/valid/",
        "scale": 8
    }
}
