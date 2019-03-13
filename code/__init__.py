import os
import logging as logger

load_pickle = True
attribute = "hgpca"
dataset_location = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset'))
clinical_path = os.path.join(dataset_location, 'All_CDEs.tsv')
meth_path = os.path.join(dataset_location, 'PRAD.meth.by_mean.data.tsv')

train_test_location = os.path.abspath(os.path.join(os.path.dirname(__file__), 'train_test_data'))
train_path = os.path.join(train_test_location, 'train_data')
test_path = os.path.join(train_test_location, 'test_data')

gene_catalog_location = os.path.abspath(os.path.join(os.path.dirname(__file__), 'gene_catalog'))
cosmic_path = os.path.join(gene_catalog_location, 'cosmic_gene_census.tsv')
civic_path = os.path.join(gene_catalog_location, 'civic_gene_summaries.tsv')

result_location = os.path.abspath(os.path.join(os.path.dirname(__file__), 'result'))
summary_path = os.path.join(result_location, 'summary.txt')

logger.basicConfig(filename=summary_path, level=logger.INFO, format='> %(message)s', filemode='a')
