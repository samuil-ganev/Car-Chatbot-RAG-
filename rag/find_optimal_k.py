import sys
import logging
import numpy as np
from pathlib import Path
import statistics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

QUESTIONS_FILE = project_root / 'questions.txt'
TOP_K_TO_ANALYZE = 50
SIMILARITY_THRESHOLD = 0.80

try:
    import search
    logging.info('Successfully imported search module')
except Exception as e:
    logging.error(f'An unexpected error occurred during import: {e}', exc_info=True)
    sys.exit(1)


def read_questions(filepath: Path) -> list[str]:
    '''
    Reads questions from a file, one per line
    '''
    
    if not filepath.is_file():
        logging.error(f'Questions file not found: {filepath}')
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        logging.info(f'Read {len(questions)} questions from {filepath}')
        return questions
    except Exception as e:
        logging.error(f'Error reading questions file {filepath}: {e}', exc_info=True)
        return []

def find_largest_score_drop_index(scores: list[float]) -> int | None:
    '''
    Calculates differences between consecutive scores and finds the index
    of the largest drop.
    '''

    scores_arr = np.array(scores)

    differences = scores_arr[:-1] - scores_arr[1:]

    if len(differences) == 0:
        logging.warning('No differences could be calculated')
        return None

    max_diff_index = np.argmax(differences)

    logging.debug(f'Scores: {scores}')
    logging.debug(f'Differences: {differences}')
    logging.debug(f'Max difference index: {max_diff_index}, Value: {differences[max_diff_index]:.4f}')

    return int(max_diff_index)


def main():
    logging.info('--- Starting Optimal K Analysis ---')

    questions = read_questions(QUESTIONS_FILE)
    if not questions:
        logging.error('No questions loaded. Exiting')
        sys.exit(1)

    logging.info('Initializing embedding model...')
    embed_model = search.initialize_embed_model(
        model_name=search.EMBEDDING_MODEL_NAME,
        device=search.EMBED_DEVICE,
        model_kwargs=search.EMBED_MODEL_KWARGS
    )
    if not embed_model:
        logging.error('Failed to initialize embedding model. Exiting')
        sys.exit(1)

    logging.info('Loading vector index...')
    index = search.load_vector_index(embed_model=embed_model)
    if not index:
        logging.error('Failed to load vector index. Exiting')
        sys.exit(1)

    optimal_k_values = []

    for i, question in enumerate(questions):
        logging.info(f'\n--- Processing Question {i+1}/{len(questions)}: \'{question}\' ---')

        results = search.search_in_index(
            query_str=question,
            index=index,
            top_k=TOP_K_TO_ANALYZE
        )

        if not results:
            logging.warning(f'No results found for question: \'{question}\'')
            continue

        results = [
            node for node in results if node.score >= SIMILARITY_THRESHOLD
        ]

        scores = [node.score for node in results]
        logging.info(f'Retrieved {len(scores)} results')

        if len(scores) < 2:
            logging.warning(f'Retrieved less than 2 results ({len(scores)}), cannot calculate drop point')
            continue

        drop_index = find_largest_score_drop_index(scores)

        if drop_index is not None:
            
            suggested_k = drop_index + 1
            optimal_k_values.append(suggested_k)
            logging.info(f'Largest score drop found after index {drop_index}. Suggested k = {suggested_k}')
            
            if drop_index > 0:
                 logging.debug(f'  Score before drop (index {drop_index-1}): {scores[drop_index-1]:.4f}')
            logging.debug(f'  Score at drop start (index {drop_index}): {scores[drop_index]:.4f}')
            if drop_index + 1 < len(scores):
                logging.debug(f'  Score after drop (index {drop_index+1}): {scores[drop_index+1]:.4f}')
            logging.debug(f'  Difference: {scores[drop_index] - scores[drop_index+1]:.4f}')

        else:
            logging.warning(f'Could not determine a drop index for question: \'{question}\'')

    if not optimal_k_values:
        logging.error('Could not determine optimal k for any question')
        final_average_k = 'N/A'
    else:
        average_k = statistics.mean(optimal_k_values)
        final_average_k = round(average_k)
        logging.info('\n--- Analysis Complete ---')
        logging.info(f'Calculated suggested k values: {optimal_k_values}')
        logging.info(f'Average suggested k (float): {average_k:.2f}')
        logging.info(f'Final Recommended k (rounded average): {final_average_k}')

    print(f'\n=============================================')
    print(f' Optimal K Analysis Results')
    print(f'=============================================')
    print(f' Number of questions processed: {len(questions)}')
    print(f' Number of questions with valid drop points: {len(optimal_k_values)}')
    print(f' List of suggested k values: {optimal_k_values}')
    print(f'---------------------------------------------')
    print(f' Recommended value for top_k (average rounded): {final_average_k}')
    print(f'=============================================')

    return final_average_k


if __name__ == '__main__':
    main()
