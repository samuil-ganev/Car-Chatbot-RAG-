'''Module for analyzing optimal `top_k` values in a vector search index'''
import sys
import logging
import statistics
from pathlib import Path
import numpy as np
import search

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

QUESTIONS_FILE = project_root / 'questions.txt'
TOP_K_TO_ANALYZE = 50
SIMILARITY_THRESHOLD = 0.80

try:
    logging.info('Successfully imported search module')
except Exception as e:
    logging.error('An unexpected error occurred during import: %s', e, exc_info=True)
    sys.exit(1)


def read_questions(filepath: Path) -> list[str]:
    '''
    Reads questions from a file, one per line
    '''
    if not filepath.is_file():
        logging.error('Questions file not found: %s', filepath)
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        logging.info('Read %d questions from %s', len(questions), filepath)
        return questions
    except Exception as e:
        logging.error('Error reading questions file %s: %s', filepath, e, exc_info=True)
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

    logging.debug('Scores: %s', scores)
    logging.debug('Differences: %s', differences.tolist())
    logging.debug('Max difference index: %d, Value: %.4f', max_diff_index,
                   differences[max_diff_index])

    return int(max_diff_index)


def main():
    '''Executes the analysis to determine the optimal value of `top_k`.
       Outputs and logs suggested values.'''
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
        logging.info('\n--- Processing Question %d/%d: \'%s\' ---', i + 1, len(questions), question)

        results = search.search_in_index(
            query_str=question,
            index=index,
            top_k=TOP_K_TO_ANALYZE
        )

        if not results:
            logging.warning('No results found for question: \'%s\'', question)
            continue

        results = [node for node in results if node.score >= SIMILARITY_THRESHOLD]
        scores = [node.score for node in results]
        logging.info('Retrieved %d results', len(scores))

        if len(scores) < 2:
            logging.warning('Retrieved less than 2 results (%d), cannot calculate drop point',
                             len(scores))
            continue

        drop_index = find_largest_score_drop_index(scores)

        if drop_index is not None:
            suggested_k = drop_index + 1
            optimal_k_values.append(suggested_k)
            logging.info('Largest score drop found after index %d. Suggested k = %d',
                          drop_index, suggested_k)

            if drop_index > 0:
                logging.debug('  Score before drop (index %d): %.4f',
                               drop_index - 1, scores[drop_index - 1])
            logging.debug('  Score at drop start (index %d): %.4f', drop_index, scores[drop_index])
            if drop_index + 1 < len(scores):
                logging.debug('  Score after drop (index %d): %.4f',
                               drop_index + 1, scores[drop_index + 1])
            logging.debug('  Difference: %.4f', scores[drop_index] - scores[drop_index + 1])

        else:
            logging.warning('Could not determine a drop index for question: \'%s\'', question)

    if not optimal_k_values:
        logging.error('Could not determine optimal k for any question')
        final_average_k = 'N/A'
    else:
        average_k = statistics.mean(optimal_k_values)
        final_average_k = round(average_k)
        logging.info('\n--- Analysis Complete ---')
        logging.info('Calculated suggested k values: %s', optimal_k_values)
        logging.info('Average suggested k (float): %.2f', average_k)
        logging.info('Final Recommended k (rounded average): %d', final_average_k)

    print('\n=============================================')
    print(' Optimal K Analysis Results')
    print('=============================================')
    print(f' Number of questions processed: {len(questions)}')
    print(f' Number of questions with valid drop points: {len(optimal_k_values)}')
    print(f' List of suggested k values: {optimal_k_values}')
    print('---------------------------------------------')
    print(f' Recommended value for top_k (average rounded): {final_average_k}')
    print('=============================================')

    return final_average_k


if __name__ == '__main__':
    main()
