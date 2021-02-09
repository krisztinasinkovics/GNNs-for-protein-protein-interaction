Exploration of Graph Attention Network architectures and Graph Convolution architectures 
to classify nodes in Protein-Protein Interaction dataset.  
Implemented in pytorch.  

### How to run:
1. Install the requirements stated in the `requirements.txt`
2. To run the train script use the following command:
```
python train.py --model_type=<'GAT' for Graph Attention Network, 'GCN' for Graph Convolution network architecture>  
                   --input_dir=<directory containing input data, e.g. 'data/ppi_dataset'>  
                    --output_dir=<dir to save the model, e.g. 'saved_models'>
```  
3. To evaluate a model on the test set use the following command:
```
python evaluate.py --model_type=<'GAT' for Graph Attention Network, 'GCN' for Graph Convolution network architecture>  
                   --input_dir=<directory containing input data, e.g. 'data/ppi_dataset'>  
                   --model_path=<directory of the model file, e.g.'saved_models/best_model_GAT.pt'>
```