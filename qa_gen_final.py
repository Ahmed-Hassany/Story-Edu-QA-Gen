"""
Question-Answer Generation System
This script implements a system that extracts events from story sections and generates
question-answer pairs based on these events.
"""

# --- Imports ---
import pandas as pd
import csv
import dspy
from dspy.datasets import DataLoader
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.teleprompt import BootstrapFewShot
from typing import Literal
from bert_score import score
from rouge_score import rouge_scorer
import numpy as np

# --- Load Datasets ---
train_df = pd.read_csv('./dataset/updated_train.csv')
val_df = pd.read_csv('./dataset/updated_valid.csv')
test_df = pd.read_csv('./dataset/updated_test.csv')

# --- Configure Language Models ---
# Select one of the following models to use

# Using llama3.1 (8B) model
lm1 = dspy.LM('ollama_chat/llama3.1', api_base='http://localhost:11434', api_key='', cache=False)
# # Using gpt-4o-mini model
# lm1 = dspy.LM('openai/gpt-4o-mini', max_tokens=3000, api_key=OPENAI_API)

dspy.configure(lm=lm1)

# --- Preparing the training data ---
# Group by 'story_section' and aggregate the required columns into lists
grouped_train_df = train_df.groupby('story_section').agg({
    'question': list,
    'answer': list,
    'event_summary': list,
    'attribute': list
}).reset_index()

event_extraction_train_examples = []
for i, row in grouped_train_df.iterrows():
    example = dspy.Example(
        story_section=row['story_section'],
        # questions=row['question'],
        # answers=row['answer'],
        event_summaries=row['event_summary'],
        # attributes=row['attribute']
    ).with_inputs('story_section')
    event_extraction_train_examples.append(example)

print(len(event_extraction_train_examples))
print(event_extraction_train_examples[0])

qa_generation_train_examples = []
for i, row in train_df.iterrows():
    example = dspy.Example(
        # story_section=row['story_section'],
        question=row['question'],
        answer=row['answer'],
        sentence=row['event_summary'],
        question_type=row['attribute']
    ).with_inputs('sentence', 'question_type')
    qa_generation_train_examples.append(example)

print(len(qa_generation_train_examples))
print(qa_generation_train_examples[0])

# --- Events Extraction ---
class ExtractEvents(dspy.Signature):
    """Extract the key events from the story section, ensuring each event is meaningful enough to inspire an educational question that encourages critical thinking and deeper comprehension for children. Extract as many significant events as possible while maintaining their quality and relevance"""
    story_section: str = dspy.InputField(prefix="Story section:", desc="The story section from which to extract events.")
    event_summaries: list[str] = dspy.OutputField(prefix="Extracted events:",desc="List of key events from the story section.")

class ExtractEventsProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.event_summaries = dspy.ChainOfThought(ExtractEvents)

    def forward(self, story_section):
        return self.event_summaries(story_section=story_section)

# --- Events Extraction Optimization ---
# Run only once to optimize the program

# import time
def bert_score_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None):
    refs = example.event_summaries
    preds = prediction.event_summaries
    # merge the references into one string
    refs = [' '.join(refs)]
    # merge the predictions into one string
    preds = [' '.join(preds)]
    # Calculate the BERT score
    _, _, F1 = score(preds, refs, lang='en', verbose=True)
    # delay 7 seconds
    # time.sleep(5)
    print(F1.mean().item())
    return F1.mean().item()

bfrs_optimizer = BootstrapFewShotWithRandomSearch(
    metric=bert_score_metric,
    max_bootstrapped_demos=2, 
    # max_rounds=1,  
    num_candidate_programs=4,
)

# Uncomment to run optimization
# optimized_events_extractor = bfrs_optimizer.compile(
#     eventExtractionProgram,
#     # teacher=teacher_classify,
#     trainset=event_extraction_train_examples[:70],
#     # valset=event_extraction_val_examples[:10],
# )

# --- Saving the compiled program ---
# save_path = './extract_events_program_optimized.json'
# optimized_events_extractor.save(save_path, save_program=False)

# save_path = './extract_events_program_optimized.pkl'
# optimized_events_extractor.save(save_path, save_program=False)

# --- Loading the compiled optimized program ---
eventExtractionProgram = ExtractEventsProgram()
eventExtractionProgram.load('./extract_events_program_optimized.json')

# --- Question Type Classification ---
lm2 = dspy.LM('ollama_chat/Llama-3.1-8B-Q-Type-Predictor', api_base='http://localhost:11434', api_key='', cache=False)

q_types = ['action', 'causal relationship', 'feeling', 'character', 'outcome resolution', 'setting']

q_type_mapping = {
    q_types[0]: "Action (A question about specific actions taken by characters in the story.)",
    q_types[1]: "Causal Relationship (A question about the reasons behind a character's actions or events in the story.)",
    q_types[2]: "Feeling (A question about a character's emotions in a given situation.)",
    q_types[3]: "Character (A question about a character in the story, their traits, or their role.)",
    q_types[4]: "Outcome Resolution (A question about the consequences or results of actions in the story, such as what happens next.)",
    q_types[5]: "Setting (A question about when and where the story takes place.)",
}

# --- Question-Answer Generation ---
class GenerateQA(dspy.Signature):
    """Convert the following declarative sentence into a relevant question-answer pair based on the following question type. The question should be directly related to the sentence's content, and the answer must be explicitly present in the sentence. """
    sentence: str = dspy.InputField(prefix="The sentence: ")
    question_type: str = dspy.InputField(prefix="Question type: ", desc="The type of question that can be asked about the sentence.")
    question: str = dspy.OutputField(prefix="Question: ", desc="A question derived from the declarative sentence.")
    answer: str = dspy.OutputField(prefix="Answer: ", desc="An answer to the generated question, explicitly present in the original sentence.")

class GenerateQAProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_qa = dspy.ChainOfThought(GenerateQA)

    def forward(self, sentence, question_type):
        return self.generate_qa(sentence=sentence, question_type=question_type)
    
generateQAProgram = GenerateQAProgram()

# --- Implementation and Testing ---

# --- Preparing the test data ---
# Group the test dataset by 'story_section' and aggregate the required columns into lists
grouped_test_df = test_df.groupby('story_section').agg({
    'question': list,
    'answer': list,
    'event_summary': list,
    'attribute': list
}).reset_index()

# include only the elements in grouped_test_df where the number of events is less than 6 and more than 1
grouped_test_df = grouped_test_df[grouped_test_df['event_summary'].apply(lambda x: len(x) < 6 and len(x)>1)]
grouped_test_df

# --- Implementation ---
generated_questions_answers = []
ignored_indices = []
ignored_events = []
ignored_qtype = []
for i, row in grouped_test_df.iterrows():
    # dspy.configure(lm=lm1)
    # optimized_events_extractor or  eventExtractionProgram (loaded from file)
    extracted_events = eventExtractionProgram(row['story_section'])
    new_item = {'story_section': row['story_section'], 'event_summaries': extracted_events.event_summaries, 'questions': [], 'answers': [], 'attributes':[]}
    print(extracted_events.event_summaries)
    for event in extracted_events.event_summaries:
        # dspy.configure(lm=lm2)
        print("Event:"+str(event))
        # q_type = classifyQTypeProgram(event)
        q_type = lm2("Classify the type of question that can be asked about the following sentence. Choose from these categories: ['action', 'causal relationship', 'character', 'setting', 'outcome resolution', 'feeling']. The sentence: "+ str(event))
        print(q_type)
        # while(q_type[0] is not in ['action', 'causal relationship','feeling','character','outcome resolution','setting'])
        if q_type[0] not in q_types:
            # ignore_flag = True
            ignored_indices.append(i)
            ignored_events.append(event)
            ignored_qtype.append(q_type[0])
        else:
            new_item['attributes'].append(q_type[0])
            # dspy.configure(lm=lm1)
            qa = generateQAProgram(event, q_type_mapping.get(q_type[0]))
            print(qa)
            new_item['questions'].append(qa.question)
            new_item['answers'].append(qa.answer)
        print("finish", i)
        print("++++++++++++++++++++++++++++++++++++++")
    generated_questions_answers.append(new_item)
print(len(generated_questions_answers))
print(generated_questions_answers[0])

print(len(generated_questions_answers))
print(len(ignored_indices))
print(len(ignored_events))
print(ignored_indices)
print(ignored_events)
print(ignored_qtype)

# save the generated_questions_answers to a csv file
generated_questions_answers_df = pd.DataFrame(generated_questions_answers)
generated_questions_answers_df.to_csv('./generated_questions_answers.csv', index=False)

# --- Evaluation ---
# Load the generated questions and answers from csv
generated_qa_df = pd.read_csv('./generated_questions_answers.csv')

# Initialize the ROUGE scorer with the metrics we want to use
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Prepare lists to store scores
rouge_scores = []

fileNameExtension = "_11_3_2025_179_not_optimized"


# Iterate through each row in the grouped_test_df
for i, row in grouped_test_df.iterrows():
    print(f"Processing row {i+1}/{len(grouped_test_df)}")
    # Extract reference questions and answers
    ref_questions = row['question']
    ref_answers = row['answer']
    
    # Combine questions and answers into a single reference text
    reference_text = ' '.join(ref_questions) + ' ' + ' '.join(ref_answers)
    
    # Get the corresponding generated questions and answers from generated_qa_df
    if i < len(generated_qa_df):
        gen_qa = generated_qa_df.iloc[i]
        gen_questions = gen_qa['questions']
        gen_answers = gen_qa['answers']
        
        # Convert to string if they are lists (eval if they are string representations of lists)
        if isinstance(gen_questions, str) and gen_questions.startswith('['):
            gen_questions = eval(gen_questions)
        if isinstance(gen_answers, str) and gen_answers.startswith('['):
            gen_answers = eval(gen_answers)
        
        # Combine into a single generated text
        generated_text = ' '.join(gen_questions) + ' ' + ' '.join(gen_answers)
        
        # Calculate ROUGE scores
        scores = scorer.score(reference_text, generated_text)

        print(f"Scores for row {i+1}:")
        print(f"ROUGE-1 P/R/F: {scores['rouge1'].precision:.4f}/{scores['rouge1'].recall:.4f}/{scores['rouge1'].fmeasure:.4f}")
        print(f"ROUGE-2 P/R/F: {scores['rouge2'].precision:.4f}/{scores['rouge2'].recall:.4f}/{scores['rouge2'].fmeasure:.4f}")
        print(f"ROUGE-L P/R/F: {scores['rougeL'].precision:.4f}/{scores['rougeL'].recall:.4f}/{scores['rougeL'].fmeasure:.4f}")
        print("===================================")
        
        # Store all metrics (precision, recall, F1) for all three ROUGE types
        rouge_scores.append({
            'rouge1': {
                'precision': scores['rouge1'].precision,
                'recall': scores['rouge1'].recall,
                'f1': scores['rouge1'].fmeasure
            },
            'rouge2': {
                'precision': scores['rouge2'].precision,
                'recall': scores['rouge2'].recall,
                'f1': scores['rouge2'].fmeasure
            },
            'rougeL': {
                'precision': scores['rougeL'].precision,
                'recall': scores['rougeL'].recall,
                'f1': scores['rougeL'].fmeasure
            }
        })

# Calculate average scores
avg_rouge1_precision = sum([score['rouge1']['precision'] for score in rouge_scores]) / len(rouge_scores)
avg_rouge1_recall = sum([score['rouge1']['recall'] for score in rouge_scores]) / len(rouge_scores)
avg_rouge1_f1 = sum([score['rouge1']['f1'] for score in rouge_scores]) / len(rouge_scores)

avg_rouge2_precision = sum([score['rouge2']['precision'] for score in rouge_scores]) / len(rouge_scores)
avg_rouge2_recall = sum([score['rouge2']['recall'] for score in rouge_scores]) / len(rouge_scores)
avg_rouge2_f1 = sum([score['rouge2']['f1'] for score in rouge_scores]) / len(rouge_scores)

avg_rougeL_precision = sum([score['rougeL']['precision'] for score in rouge_scores]) / len(rouge_scores)
avg_rougeL_recall = sum([score['rougeL']['recall'] for score in rouge_scores]) / len(rouge_scores)
avg_rougeL_f1 = sum([score['rougeL']['f1'] for score in rouge_scores]) / len(rouge_scores)

# Print average scores
print("\nAverage ROUGE Scores:")
print(f"ROUGE-1 P/R/F: {avg_rouge1_precision:.4f}/{avg_rouge1_recall:.4f}/{avg_rouge1_f1:.4f}")
print(f"ROUGE-2 P/R/F: {avg_rouge2_precision:.4f}/{avg_rouge2_recall:.4f}/{avg_rouge2_f1:.4f}")
print(f"ROUGE-L P/R/F: {avg_rougeL_precision:.4f}/{avg_rougeL_recall:.4f}/{avg_rougeL_f1:.4f}")


# Save scores to separate files
rouge1_df = pd.DataFrame([{'Precision': avg_rouge1_precision, 'Recall': avg_rouge1_recall, 'F1': avg_rouge1_f1}])
rouge1_df.to_csv(f'rouge1{fileNameExtension}.csv', index=False)
rouge2_df = pd.DataFrame([{'Precision': avg_rouge2_precision, 'Recall': avg_rouge2_recall, 'F1': avg_rouge2_f1}])
rouge2_df.to_csv(f'rouge2{fileNameExtension}.csv', index=False)
rougeL_df = pd.DataFrame([{'Precision': avg_rougeL_precision, 'Recall': avg_rougeL_recall, 'F1': avg_rougeL_f1}])
rougeL_df.to_csv(f'rougeL{fileNameExtension}.csv', index=False)

# Save the scores (Not just the average) to separate files
rouge1_scores_df = pd.DataFrame([score['rouge1'] for score in rouge_scores])
rouge1_scores_df.to_csv(f'rouge1_scores{fileNameExtension}.csv', index=False)
rouge2_scores_df = pd.DataFrame([score['rouge2'] for score in rouge_scores])
rouge2_scores_df.to_csv(f'rouge2_scores{fileNameExtension}.csv', index=False)
rougeL_scores_df = pd.DataFrame([score['rougeL'] for score in rouge_scores])
rougeL_scores_df.to_csv(f'rougeL_scores{fileNameExtension}.csv', index=False)


print("\nResults have been saved to CSV files.")

# Calculate the BERTScore for generated questions and answers compared to reference QA pairs

# Prepare lists to store scores
bert_scores = []

# Define a file name extension for the output files (optional - using same format as ROUGE evaluation)
fileNameExtension = "_11_3_2025_179_not_optimized"

# Iterate through each row in the grouped_test_df
print("Calculating BERTScores...")
for i, row in grouped_test_df.iterrows():
    print(f"Processing row {i+1}/{len(grouped_test_df)}")
    
    # Extract reference questions and answers
    ref_questions = row['question']
    ref_answers = row['answer']
    
    # Combine questions and answers into a single reference text
    reference_text = ' '.join(ref_questions) + ' ' + ' '.join(ref_answers)
    references = [reference_text]
    
    # Get the corresponding generated questions and answers
    if i < len(generated_qa_df):
        gen_qa = generated_qa_df.iloc[i]
        gen_questions = gen_qa['questions']
        gen_answers = gen_qa['answers']
        
        # Convert to list if they are string representations of lists
        if isinstance(gen_questions, str) and gen_questions.startswith('['):
            gen_questions = eval(gen_questions)
        if isinstance(gen_answers, str) and gen_answers.startswith('['):
            gen_answers = eval(gen_answers)
        
        # Combine into a single generated text
        generated_text = ' '.join(gen_questions) + ' ' + ' '.join(gen_answers)
        candidates = [generated_text]
        
        # Calculate BERTScore (measures semantic similarity between generated and reference text)
        P, R, F1 = score(candidates, references, lang='en', verbose=True)
        
        print(f"BERTScore P/R/F: {P.item():.4f}/{R.item():.4f}/{F1.item():.4f}")
        print("===================================")
        
        # Store scores
        bert_scores.append({
            'precision': P.item(),
            'recall': R.item(),
            'f1': F1.item()
        })

# Calculate average scores
avg_precision = np.mean([score['precision'] for score in bert_scores])
avg_recall = np.mean([score['recall'] for score in bert_scores])
avg_f1 = np.mean([score['f1'] for score in bert_scores])

# Print average scores
print("\nAverage BERTScores:")
print(f"Precision: {avg_precision:.4f}")
print(f"Recall: {avg_recall:.4f}")
print(f"F1: {avg_f1:.4f}")

# Save the results to CSV files
results_df = pd.DataFrame([{'Precision': avg_precision, 'Recall': avg_recall, 'F1': avg_f1}])
results_df.to_csv(f'bertscore_results{fileNameExtension}.csv', index=False)

# Save all individual scores
detailed_scores_df = pd.DataFrame(bert_scores)
detailed_scores_df.to_csv(f'bertscore_detailed{fileNameExtension}.csv', index=False)

print("\nResults have been saved to CSV files.")
