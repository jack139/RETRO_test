from configs import TRAIN_DATA_PATH, SEQ_LEN, CHUNK_SIZE, NUM_NEIGHBORS
from retro_pytorch import retrieval

retrieval.TMP_PATH = retrieval.Path(f'{TRAIN_DATA_PATH}/tmp')
retrieval.INDEX_FOLDER_PATH = retrieval.TMP_PATH / '.index'


stats = retrieval.text_folder_to_chunks_(
    folder = './test_data/text',
    glob = '**/wiki*.txt',
    chunks_memmap_path = f'{TRAIN_DATA_PATH}/train.chunks.dat',
    seqs_memmap_path = f'{TRAIN_DATA_PATH}/train.seq.dat',
    doc_ids_memmap_path = f'{TRAIN_DATA_PATH}/train.doc_ids.dat',  # document ids are needed for filtering out neighbors belonging to same document appropriately during computation of nearest neighbors
    chunk_size = CHUNK_SIZE,
    seq_len = SEQ_LEN,
    max_chunks = 1_000_000,
    max_seqs = 100_000
)

NUM_CHUNKS, NUM_DOCS, NUM_SEQS = stats['chunks'], stats['docs'], stats['seqs']

print('NUM_CHUNKS = %d\tNUM_DOCS = %d\tNUM_SEQS = %d'%(NUM_CHUNKS, NUM_DOCS, NUM_SEQS))


retrieval.chunks_to_precalculated_knn_(
    num_chunks = NUM_CHUNKS,
    chunk_size = CHUNK_SIZE,
    chunk_memmap_path = f'{TRAIN_DATA_PATH}/train.chunks.dat',    # path to main chunks dataset
    doc_ids_memmap_path = f'{TRAIN_DATA_PATH}/train.doc_ids.dat', # path to document ids created by text_folder_to_chunks_, used for filtering out neighbors that belong to the same document
    num_nearest_neighbors = NUM_NEIGHBORS,                   # number of nearest neighbors you'd like to use
    num_extra_neighbors = 10                     # fetch 10 extra neighbors, in the case that fetched neighbors are frequently from same document (filtered out)
)


'''
from retro_pytorch.retrieval import chunks_to_index_and_embed

index, embeddings = chunks_to_index_and_embed(
    num_chunks = 1000,
    chunk_size = 64,
    chunk_memmap_path = './test_data/train.chunks.dat'
)

query_vector = embeddings[:1]                   # use first embedding as query
_, indices = index.search(query_vector, k = 2)  # fetch 2 neighbors, first indices should be self

neighbor_embeddings = embeddings[indices]       # (1, 2, 768)
'''

