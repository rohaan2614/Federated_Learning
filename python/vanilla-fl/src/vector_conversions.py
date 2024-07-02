import logging
import torch
from models import CNNMnist
import json

def main(RANDOM_SEED: int = 2,
         ):
    
    torch.manual_seed(RANDOM_SEED)

    # Check if CUDA is available, if not use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Initialize the global model
    global_model = CNNMnist().to(device)
    logging.info("Global model initialized.")

    # Copy global weights
    global_weights = global_model.state_dict()
    # Convert global weights to a JSON-compatible format
    global_weights_json = {k: v.tolist() for k, v in global_weights.items()}
    # Write global weights to a JSON file
    with open('global_weights_original.json', 'w') as f:
        json.dump(global_weights_json, f)
        
    # Flatten all parameters into a single vector
    parameters_vector = global_model.weights_to_vector()
    # Save vector to JSON
    with open('flat_vectors.json', 'w') as f:
        json.dump(parameters_vector.tolist(), f)
    print("Model parameters saved to 'flat_vectors.json'")
    
    global_model.zero_out_weights()
    # Confirm weights are zero
    global_weights = global_model.state_dict()
    global_weights_json = {k: v.tolist() for k, v in global_weights.items()}
    with open('global_weights_interim.json', 'w') as f:
        json.dump(global_weights_json, f)
    
    # Load vector back into model
    with open('flat_vectors.json', 'r') as f:
        loaded_vector = torch.tensor(json.load(f))
    global_model.vector_to_weights(loaded_vector)
    print("Model parameters loaded from 'flat_vectors.json'")

    # Confirm final weights
    global_weights = global_model.state_dict()
    global_weights_json = {k: v.tolist() for k, v in global_weights.items()}
    with open('global_weights_final.json', 'w') as f:
        json.dump(global_weights_json, f)

if __name__ == '__main__':
    main()
