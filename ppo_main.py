def main():
    # TODO: Setup argument parser 
    parser = argparse.ArgumentParser()
    # parser.add_argument('--<NAME>', type=<TYPE>)
    # ...
    args = parser.parse_args()

    params = vars(args)

    # Set random seeds
    set_random_seeds()

    # Start training loop
    trainer = PPO_trainer(params)
    trainer.run_training_loop()

if __name__ == "__main__":
    main()