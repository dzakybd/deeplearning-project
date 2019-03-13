import pandas as pd
from __init__ import *
import warnings
warnings.simplefilter(action='ignore')

def cosmic_cancer_genes():
    """
    Read COSMIC Cancer Gene Census - catalogue those genes for which mutations have been causally implicated in cancer
    """
    gene_census_data = pd.read_csv(cosmic_path, skipinitialspace=True, usecols=['Gene Symbol', 'Synonyms'], delimiter='\t')
    gene_census = list(gene_census_data['Gene Symbol'])

    for synonynm in gene_census_data['Synonyms']:
        if type(synonynm) is str:
            gene_census.extend(synonynm.split(','))
    logger.info('Number of COSMIC Cancer Catalogue: {0}\n'.format(len(gene_census)))
    return gene_census


def civic_cancer_genes():
    """
    Read Clinical Interpretation of Variants in Cancer (Civic) catalogue of cancer genes
    """
    civic_genes_data = pd.read_csv(civic_path, skipinitialspace=True, usecols=['name'], delimiter='\t')
    civic_genes = list(civic_genes_data['name'])
    logger.info('Number of Civic Cancer Catalogue: {0}\n'.format(len(civic_genes)))
    return civic_genes


def genes_feature_selection(methyl_data, cancer_genes):
    """
    Reduce feature space of protein-binding genes by considering COSMIC & CIVIC data
    """
    overlap_genes = cancer_genes.intersection(methyl_data.index)
    return methyl_data.ix[overlap_genes]


def miss_data_impute(methyl_data):
    for col in methyl_data.columns:
        methyl_data[col] = (pd.to_numeric(methyl_data[col], errors='coerce'))
        methyl_data[col] = methyl_data[col].fillna(methyl_data[col].mean())
    return methyl_data


def add_label(methyl_data):
    case_ids = methyl_data.columns.values
    labels = []

    clinical_data = pd.read_csv(clinical_path, skipinitialspace=True, usecols=['bcr_patient_barcode', 'gleason_score'], delimiter='\t')
    clinical_dic = dict(zip(clinical_data['bcr_patient_barcode'], clinical_data['gleason_score']))
    for case_id in case_ids:
        short_id = '-'.join(case_id.split('-')[:-1]).lower()
        gleason_score = clinical_dic.get(short_id)
        if gleason_score <= 7:
            labels.append(0)
        else:
            labels.append(1)

    labels = [int(i) for i in labels]
    methyl_data.loc[methyl_data.shape[0]] = labels
    methyl_data = methyl_data.rename(index={methyl_data.shape[0] - 1: attribute})

    return methyl_data


def train_test_data(methyl_data):
    training_data = pd.DataFrame()
    testing_data = pd.DataFrame()

    not_severe_group = methyl_data[methyl_data[attribute] == 0]
    logger.info("not_severe_group.shape " + str(not_severe_group.shape))
    train_ns_group = not_severe_group.sample(frac=0.7, random_state=1)
    test_ns_group = not_severe_group.drop(train_ns_group.index)

    severe_group = methyl_data[methyl_data[attribute] == 1]
    logger.info("severe_group.shape " + str(severe_group.shape))
    train_s_group = severe_group.sample(frac=0.7, random_state=1)
    test_s_group = severe_group.drop(train_s_group.index)

    training_data = training_data.append(train_ns_group)
    training_data = training_data.append(train_s_group)
    testing_data = testing_data.append(test_ns_group)
    testing_data = testing_data.append(test_s_group)

    training_data = training_data.sample(frac=1, random_state=1)
    testing_data = testing_data.sample(frac=1, random_state=1)

    logger.info('Saving training and testing data')
    training_data.to_csv(train_path+".csv")
    testing_data.to_csv(test_path+".csv")
    training_data.to_pickle(train_path+".pkl")
    testing_data.to_pickle(test_path+".pkl")

    logger.info('Processing completed!')
    return training_data, testing_data


def build():
    """
    Read and process DNA methylation data and preprocessing - feature selection, missing data imputation & adding label
    """
    logger.info('Process initiated - Building dataset')

    if os.path.isfile(train_path) and os.path.isfile(test_path) and load_pickle:
        logger.info('Loading train & test data')
        return pd.read_pickle(train_path+".pkl"), pd.read_pickle(test_path+".pkl")

    logger.info('Reading COSMIC Cancer Gene Census')
    gene_census = cosmic_cancer_genes()
    logger.info('Reading CIVIC Cancer Gene Census')
    gene_census.extend(civic_cancer_genes())
    gene_census = set(gene_census)
    gene_census = {x for x in gene_census if pd.notna(x)}

    logger.info('Reading Methylation data of TCGA PRAD')
    methyl_data = pd.read_csv(meth_path, delimiter='\t', skiprows=[1], index_col=0)
    logger.info('Number of Genes: {0} | Number of Patients: {1}'.format(methyl_data.shape[0], methyl_data.shape[1]))
    logger.info('Preprocessing Methylation data')

    methyl_data = genes_feature_selection(methyl_data, gene_census)
    logger.info('Number of Genes after processing: {0}\n'.format(methyl_data.shape[0]))

    methyl_data = miss_data_impute(methyl_data)
    methyl_data = add_label(methyl_data)
    methyl_data = methyl_data.transpose()

    return train_test_data(methyl_data)

if __name__ == '__main__':
    build()
