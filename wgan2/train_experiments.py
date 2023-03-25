

from train import default_config, train


if __name__ == "__main__":
    config = default_config()

    for kernel_size in [(8,8,8),(16,16,16),(32,32,32),(64,64,64)]:

        config["training_parameters"]["kernel_size"] = kernel_size
        config["discriminator_config"]["spatial_size"] = kernel_size

        for low_res_size in [(80, 384, 384), (40, 384, 384), (20, 384, 384)]:
            config["training_parameters"]["low_res_size"] = low_res_size

            for n_dense_blocks in [3,5,8]:
                config["generator_config"]["n_dense_blocks"] = n_dense_blocks
                
                train(config)
     

