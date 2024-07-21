import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from rdkit import Chem
from rdkit.Chem import Descriptors, PandasTools, QED

# Load known actives
known_actives_df = pd.read_csv('../actives.csv')
PandasTools.AddMoleculeColumnToFrame(known_actives_df, smilesCol='smiles')

# Generate properties for known actives
def compute_properties(mol):
    if mol is None:
        return {}
    return {
        'MolWt': Descriptors.MolWt(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'RingCount': Descriptors.RingCount(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'QED': QED.qed(mol)
    }

known_actives_df = known_actives_df.dropna(subset=['ROMol'])  # Drop rows where ROMol is NaN
properties = known_actives_df['ROMol'].apply(compute_properties)
properties_df = pd.DataFrame(list(properties))
known_actives_df = pd.concat([known_actives_df, properties_df], axis=1)

# Standardize the properties
scaler = StandardScaler()
X_scaled = scaler.fit_transform(properties_df)

# Perform PCA
pca = PCA(n_components=2)  # For visualization, use 2 components
X_pca = pca.fit_transform(X_scaled)

# Add PCA components to DataFrame
known_actives_df['PCA1'] = X_pca[:, 0]
known_actives_df['PCA2'] = X_pca[:, 1]

# Plotting
# 1. Distribution of Key Properties for the known actives
plt.figure(figsize=(14, 14))
for i, column in enumerate(properties_df.columns, 1):
    plt.subplot(4, 3, i)
    sns.histplot(known_actives_df[column], kde=True)
    #plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.savefig('actives_distribution_of_properties.png')


# Comparison of Active vs new_molecules Compounds
known_actives_df['activity'] = 'Active'  # Assuming all known compounds are active

# Load new compounds and compute properties
new_compounds_df = pd.read_csv('top_hits/concatenated/enamine_29-38.csv')  # Assuming CSV with a column 'smiles'
PandasTools.AddMoleculeColumnToFrame(new_compounds_df, smilesCol='smiles')
new_compounds_df = new_compounds_df.dropna(subset=['ROMol'])  # Drop rows where ROMol is NaN
new_properties = new_compounds_df['ROMol'].apply(compute_properties)
new_properties_df = pd.DataFrame(list(new_properties))
new_compounds_df = pd.concat([new_compounds_df, new_properties_df], axis=1)
new_compounds_df['activity'] = 'new_molecules'

# Ensure 'smiles' and 'idnumber' columns are present in new_compounds_df
required_columns = ['smiles', 'idnumber']
for col in required_columns:
    if col not in new_compounds_df.columns:
        new_compounds_df[col] = np.nan  # Add missing columns with NaN values


# Align columns for combination
properties_columns = properties_df.columns.tolist()
known_actives_df = known_actives_df[['activity'] + properties_columns + ['PCA1', 'PCA2']]
new_compounds_df = new_compounds_df[['activity'] + properties_columns + required_columns]

# Combine both datasets for comparison
combined_df = pd.concat([known_actives_df, new_compounds_df], ignore_index=True)

# Perform PCA on new compounds
X_new_scaled = scaler.transform(new_properties_df)
X_new_pca = pca.transform(X_new_scaled)
new_compounds_df['PCA1'] = X_new_pca[:, 0]
new_compounds_df['PCA2'] = X_new_pca[:, 1]

# Calculate distances between new compounds and actives in PCA space
new_molecules_pca = new_compounds_df[['PCA1', 'PCA2']].values
active_pca = known_actives_df.loc[known_actives_df['activity'] == 'Active', ['PCA1', 'PCA2']].values
distances = pairwise_distances(new_molecules_pca, active_pca, metric='euclidean')

# Calculate scores based on mean distance to actives
scores = np.mean(distances, axis=1)
new_compounds_df['Score'] = scores

# Save top hits
top_hits = new_compounds_df.nsmallest(10000, 'Score')

# Visualizations

# 1. PCA Visualize the top 10000 hits vs actives
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA1', y='PCA2', data=known_actives_df, palette='viridis', s=60, label='Active')
sns.scatterplot(x='PCA1', y='PCA2', data=top_hits, color='red', marker='X', s=80, label='Top 10000 Hits')
plt.title('Top hits vs actives in PCA Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.savefig('top_hits_pca_visualization.png')
#plt.show()

# 2. Distribution of Key Properties for Top Hits
plt.figure(figsize=(14, 14))
for i, column in enumerate(properties_columns, 1):
    plt.subplot(4, 3, i)
    sns.histplot(top_hits[column], kde=True)
    #plt.title(f'Distribution of {column} in Top Hits')
plt.tight_layout()
plt.savefig('top_hits_distribution_of_properties.png')
#plt.show()

# 3. Violin Plots for Top Hits vs Actives
plt.figure(figsize=(14, 14))
for i, column in enumerate(properties_columns, 1):
    plt.subplot(4, 3, i)
    sns.violinplot(x='activity', y=column, data=pd.concat([known_actives_df, top_hits]))
    #plt.title(f'{column}')
plt.tight_layout()
plt.savefig('violin_plots_top_hits_vs_actives.png')
#plt.show()

# Save top 100 hits to a CSV file
top_hits[['smiles', 'idnumber', 'Score']].to_csv('top_10000_hits.csv', index=False)

