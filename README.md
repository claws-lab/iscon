# Implicit Session Contexts for Next-Item Recommendations (ACM CIKM 2022 paper)

Overview
---------------
**Implicit Session Contexts for Next-Item Recommendations**  
[Sejoon Oh](https://sejoonoh.github.io/), Ankur Bhardwaj, Jongseok Han, [Sungchul Kim](https://research.adobe.com/person/sungchul-kim/), [Ryan A. Rossi](http://ryanrossi.com/), and [Srijan Kumar](https://www.cc.gatech.edu/~srijan/)  
*[ACM International Conference on Information and Knowledge Management (CIKM)](https://www.cikm2022.org/) short paper, 2022*   

Link to the paper PDF - will be ready soon.

**ISCON** is a novel session-based recommendation framework that contextualize sessions and utilizes the contextual embeddings for the next-item prediction.  
**Datasets** used for in the paper are available at the following link.  
https://drive.google.com/file/d/1Q8bCw3PqoPayP9RiKO7TaRHhKAotevkA/view?usp=sharing

Usage
---------------

The detailed execution procedure of **ISCON** is given as follows.

1) Install all required libraries by "pip install -r requirements.txt" (Python 3.6 or higher version is required).
3) "python src/main.py [arguments]" will execute ISCON with arguments, and specific information of the arguments are as follows.

````
--data_name: name of the input data (e.g., reddit). The corresponding data should be in the data directory with the file name "[data_name].tsv".
--epochs: number of epochs for training (default: 200)
--gpu: GPU number will be used for experiments (default: 0)
--output: name of the output log file (e.g., reddit_output)
--contexts: number of session contexts (default: 40)
--batch_size: mini-batch size for training (default: 1024)
--train_ratio: training/test split ratio (default: 0.9)
--session_emb_dim and --emb_dim: session embedding dimension (default: 128) and user&item embedding dimension (default: 256)
--context_dim: contextual embedding dimension (default: 32)
--topk: number of predicted contexts per session (default: 3)
````

Demo
---------------
To run the demo, please follow the following procedure. **ISCON** demo will be executed on the Reddit dataset.

	1. Check permissions of files (if not, use the command "chmod 777 *")
	2. Execute "./demo.sh"
	3. Check "output/reddit_demo" for the demo result of ISCON on the Reddit dataset
  
Tested Environment
---------------
We tested our proposed method **ISCON** in NVIDIA DGXStation machines equipped with 8 NVIDIA Tesla V100 GPUs.
