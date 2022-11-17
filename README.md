

# Automated Summarization of Stack Overflow Posts

Navigating SO posts for solutions to programming tasks and comparing solutions in different posts is a challenging task for programmers with a limited time budget. To provide tool support for fast SO post navigation, we present ASSORT: a framework for generating sentence-level extractive summaries of Stack Overflow (SO) posts.  ASSORT is consist of two components:  ASSORT<sub>S</sub>, a novel ensemble-based supervised model architecture for extractive SO post summarization, and ASSORT<sub>IS</sub>, an indirect supervision approach that requires no expensive labeled data for the same task.  In this work, we perform a quantitative analysis of the performance of ASSORT<sub>S</sub> and ASSORT<sub>IS</sub> by comparing them against 6 carefully selected baselines including a state-of-the-art transformer-based extractive summarization model. We also perform a user study to learn how real programmers perceive summaries generated by ASSORT.  Data and codes to replicate our work are included in this repository, as well as the user study results and a Chrome extension with ASSORT implemented that allows online SO post summarization.

## Data
For training and testing purposes, we expanded SOSum, the first dataset of SO post summaries, following the same procedure described in **"SOSum: A Dataset of Stack Overflow Post Summaries."**[^fn1] The dataset can be found as a csv file with the path of `data/dataset.csv`. Specifically, `dataset.csv` contains the following columns:

 - sentence: A sentence from a SO post.
 - answer_id: Id of the answer post.
 - question_id: Id of the question post.
 - answer_body: The answer post. Most HTML tags are removed for clarity but some are preserved for calculating explicit features, such as `<code>`.
 - position: The position of the sentence in the post. E.g. 1 means the first sentence in the post.
 - question_title: Title of the question, used to classify the question.
 - truth: Whether the sentence is selectd as a summative sentence or not by the human labelers.
 - tags: The SO tags associated with the question.
 - question_type: Questions are classified into three distinct categories---*how-to questions*,  *conceptual questions*, and *bug fixing questions*. How-to questions ask for instructions for achieving a task, e.g., "how do I undo the most recent local commits in Git?". Conceptual questions ask for clarifications on a concept, e.g., "what are metaclasses in Python?'". Bug fixing questions ask for solutions to fix some issues, e.g. "git is not working after macOS Update". The question type. 1=Conceptual, 2=How-to, 3=Bug-fixing
 - question_body: The question description.

In general, the entire dataset contains sentences from 3,063 SO posts under popular questions and their manually curated summaries.  Among these answer posts, 254 of them are answers to *how-to questions*, 322 to *conceptual questions*, and 209 to *bug-fixing questions*. We split the entire dataset into train/dev/test sets by the ratio of 8:1:1.

[^fn1]: Kou, Bonan, et al. "SOSum: A Dataset of Stack Overflow Post Summaries." _2022 IEEE/ACM 19th International Conference on Mining Software Repositories (MSR)_. IEEE, 2022.
## ASSORT<sub>S</sub>,
As our supervised model for summarizing SO posts, ASSORT achieves desirable performance on sentence-level extractive summarization of SO posts, outperforming a state-of-the-art extractive summarization model by 13% in terms of F1 score. ASSORT<sub>S</sub> takes three phases to summarize a SO answer post. Since answers to different types of questions follow different linguistic patterns, it first predicts the type of the SO question (Phase I). To account for the uncertainty of the question classifier, the answer post is fed into three sentence classification models separately, each of which is trained for one type of SO question (Phase II). ASSORT<sub>S</sub> predicts confidence scores with multiple sentence classifiers trained for different question categories. Finally, it ensembles the predictions of these models based on the likelihood of the question type to generate the final summary (Phase III). The figure below visualizes the three phases.
[![Supervised](https://anonymous.4open.science/r/ASSORT-Automatic-Summarization-of-Stack-Overflow-Posts-4FE3/screenshots/supervised.png "Supervised")](https://anonymous.4open.science/r/ASSORT-Automatic-Summarization-of-Stack-Overflow-Posts-4FE3/screenshots/supervised.png "Supervised")



The script for training the question classifier and sentence classifier can be found in `model` folder. For example, if you want to train a question classifier from scratch, simply run:

`python train_question_classifier.py`

By default, this file will split the train/dev/test sets with the ratio of 8:1:1 to train a question classifer which is a SVM classifier. After training is done, the script will evaluate the classifier's performance on the test set.

Similarly, to train a sentence classifier from scratch, run:

`python train_sentence_classifier.py`

Note that our supervised model get sentence embedding from a Huggingface implementation of BERT model. Therefore, users have to make sure they have the transformers package installed. You can install the package with the following command.

`pip install transformers`

## ASSORT<sub>IS</sub>
While supervised learning can achieve superior performance, obtaining a large amount of labeled data is often costly, especially in specific domains such as software engineering.
Therefore, we propose an indirect supervision approach, ASSORT<sub>IS</sub>, to overcome this limit. Instead of acquiring labeled data for direct supervision, ASSORT<sub>IS</sub> uses supervision signals from pre-trained models in another domain, such as news article summarization. To address the challenge of data shift in cross-domain transfer, we use a pre-trained Natural Language Inference (NLI) model to select summative sentences in the original post based on the summary generated by the pre-trained text summarization model. The below figure provides an overview of our approach.
[![indirect supervised](https://anonymous.4open.science/r/ASSORT-Automatic-Summarization-of-Stack-Overflow-Posts-4FE3/screenshots/indirect_supervision.png "indirect supervised")](https://anonymous.4open.science/r/ASSORT-Automatic-Summarization-of-Stack-Overflow-Posts-4FE3/screenshots/indirect_supervision.png "indirect supervised")

To apply ASSORT<sub>IS</sub> on our dataset, use script in `model/indirect_supervision.py`.

## Chrome extension
To give our users an opportunity to try ASSORT, we build a chrome extension that can summarize online SO posts. Below is a screenshot of our Chrome extension when properly installed in the browswer.

[![chrome_extension](https://anonymous.4open.science/r/ASSORT-Automatic-Summarization-of-Stack-Overflow-Posts-4FE3/screenshots/chrome.png "chrome_extension")](https://anonymous.4open.science/r/ASSORT-Automatic-Summarization-of-Stack-Overflow-Posts-4FE3/screenshots/chrome.png "chrome_extension")

Folder `chrome_extension` contains codes for the local server as well as the Chrome extension itself. To load the extension, first run `server.py` in `server` folder. Then go to chrome://extensions and upack the `extension` folder in developer mode. Then, open any Stack Overflow page, the extension should start to work.

## Experiment scripts
We perform a quantative study to compare ASSORT<sub>IS</sub> and ASSORT<sub>S</sub> with 6 popular baselines. These baselines are listed below:

- ***BERTSUM***[^fn2] is an extractive summarization model that first uses BERT to encode sentences and then uses a transformer to select summative sentences. It outperforms several previous techniques on two popular text summarization datasets---NYT. In this experiment, we use the checkpoint of \liu{} that has the best performance on CNN/DailyMail.
- ***BERTSUM (fine-tuned)*** is a fine-tuned version of BERTSUM. It is fine-tuned with the training data of the supervised ASSORT, including 2,424 SO posts and their summaries.
  
- ***wordpattern*** [^fn3] identifies essential sentences in a SO post using a set of 360 word patterns. These patterns are initially designed by Robillard and Chhetri~\cite{robillard2015recommending} to identify sentences containing indispensable knowledge in API documentations.
  
- ***simpleif*** [^fn3] is a technique proposed by Nadi and Treude. It is designed based on the insight that essential sentences may contain contextual information expressed in the form of conditions. Thus, simpleif identifies all sentences that have the word 'if' in them as essential sentences.
  
- ***contextif*** [^fn3] is another technique proposed by Nadi and Treude~\cite{essential_sentences}. It uses a set of heuristics to identify essential sentences that carry technical context and are useful.

- ***lexrank*** [^fn4] is a commonly used unsupervised text summarization approach. It uses a stochastic graph-based method to compute the relative importance of sentences in a document and generates an extractive summary by selecting the top k sentences. We use k=5.

For evaluating these baseline models on our dataset, please run script in `model/baseline.py` with the following command:
`python baseline.py`

[^fn2]: https://arxiv.org/pdf/1908.08345.pdf

[^fn3]: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9054828

[^fn4]: https://pypi.org/project/lexrank/


## User Study
In this work, we also conduct a user study with 12 graduate student in the CS department of a R1 university to study how summaries generated by ASSORT are perceived by real programmers. We form a pool of SO posts by randomly selecting 40 answer posts from our dataset, including 13 *how-to questions*, 13 *conceptual questions*, and 14 *bug-fixing questions*. In each user study, we select 10 out of the 40 answer posts and ask the participant to review their summaries. We counterbalance the post assignment so that each post is evaluated by three different participants. For each answer post, participants first report their expertise of the concepts in the question on a 7-point scale.  1 means "Haven't even heard of it before" and 7 means "I am an expert". Then, they will be provided with the question post, the answer post, and the summaries generated by ASSORT<sub>S</sub>, ASSORT<sub>IS</sub>, and BertSum (fine-tuned). Specifically, we select BertSum (fine-tuned) as our baseline in the user study since it performs the best among all six baselines in the quantitative experiment. After reading all three summaries of a post, the participants evaluate the quality of these summaries by answering the following five multiple-choice questions. Both the pool of 40 SO posts and summaries generated by three models (`user_study/user_study_pool.csv`) and the responses of the participants (`user_study/user_study_result.csv`) are made publicly available in this repository.

*Note:* Some cells in `user_study/user_study_result.csv` are left blank because the participants have skipped it during the survey.

