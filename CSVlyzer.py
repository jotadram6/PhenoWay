import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
import awkward
import mplhep as hep
import glob
hep.style.use("CMS")

class RootTreeReader:

    """ 
    Read data from a ROOT TTree 
    Parameters:
    path : string
        Path to the ROOT file
    tree_name : string (default=Delphes)
        Name of the ROOT TTree
    Attributes:
    tree: Root TTree 
    """

    def __init__(self, path: str, tree_name: str = "Delphes"):
        self.tree = uproot.open(path)[tree_name]

    def get_branches(self, branches = ["MissingET.MET",
                                       "MissingET.Phi",
                                       "Jet.PT",
                                       "Jet.Eta",
                                       "Jet.Phi",
                                       "Jet.Mass",
                                       "Jet.TauTag",
                                       "Jet.BTag",
                                       "Jet_size",
                                       "Electron.PT",
                                       "Electron.Eta",
                                       "Electron.Phi",
                                       "Electron.Charge",
                                       "Electron_size",
                                       "Muon.PT",
                                       "Muon.Eta",
                                       "Muon.Phi",
                                       "Muon.Charge",
                                       "Muon_size"], max_j_elements=4, max_l_elements=2):

        """
        returns a DataFrame with branches as features
        branches : array-like
          branches to load from the ROOT tree
        max_elements : int (default=4)
          maximum number of elements to load from jagged arrays
        """
        self._max_j_elements = max_j_elements
        self._max_l_elements = max_l_elements
        self._df = pd.DataFrame(index=range(self.tree.num_entries))

        for branch in branches:
            self._join_branch(branch)

        return self._set_columns_names(self._df)

    
    def _join_branch(self, branch):
        """joins a branch to self._df"""
        df = self.tree.arrays(branch, library="pd")
        
        if "." in branch:
            if len(df) > len(df.groupby(level=0).size()):
                if "Jet" in branch:
                    self._add_jagged_branch(df, branch, self._max_j_elements)
                if "Electron" in branch or "Muon" in branch:
                    self._add_jagged_branch(df, branch, self._max_l_elements)
            else:
                self._add_branch(df, branch)
        else:
            self._add_branch(df, branch)

            
    def _add_branch(self, df, branch: str):
        """adds a non-jagged branch to self.df"""
        self._df[branch] = self.tree[branch].array(library="pd").values


    def _add_jagged_branch(self, df, branch, max_i):
        """adds a jagged branch to self.df"""
        df = df.unstack().iloc[:,:max_i]
        df.columns = ["{0}{1}".format(branch, i) for i in range(max_i)]
        self._df = self._df.join(df)

    @staticmethod
    def _set_columns_names(df):
        df.columns = df.columns.str.lower().str.replace(".","_")
        return df


def build_df(path):
    """
    Generates a Dataframe from the root in "path"
    """
    reader = RootTreeReader(path)
    df = reader.get_branches()
    df["n_b"]  =  reader.tree.arrays("Jet.BTag", library="pd").sum(level=0)
    df["n_tau"] = reader.tree.arrays("Jet.TauTag", library="pd").sum(level=0)
    #df['n_tau'] = np.sum(df.loc[:,"jet_tautag0":"jet_tautag3"],axis = 1)
    #df['n_b'] = np.sum(df.loc[:,"jet_btag0":"jet_btag3"],axis = 1)
    return df


#WprimeZ1="/cms/mc/Samples/VBF_Wprime/Wprime_VBF_gWWpZ1.root"
#df_z1 = build_df(WprimeZ1)
#df_z1.to_csv("Wprime_VBF_gWWpZ1.csv")

#WprimeA1="/cms/mc/Samples/VBF_Wprime/Wprime_VBF_gWWpA1.root"
#df_a1 = build_df(WprimeA1)
#df_a1.to_csv("Wprime_VBF_gWWpA1.csv")

#WprimeZ2="/cms/mc/Samples/VBF_Wprime/Wprime_VBF_gWWpZ2.root"
#df_z2 = build_df(WprimeZ2)
#df_z2.to_csv("Wprime_VBF_gWWpZ2.csv")

#WprimeA2="/cms/mc/Samples/VBF_Wprime/Wprime_VBF_gWWpA2.root"
#df_a2 = build_df(WprimeA2)
#df_a2.to_csv("Wprime_VBF_gWWpA2.csv")

#WprimeZ1_M500="/cms/mc/Samples/VBF_Wprime/Wprime_VBF_gWWpZ1_M500.root"
#df_z1_M500 = build_df(WprimeZ1_M500)
#df_z1_M500.to_csv("Wprime_VBF_gWWpZ1_M500.csv")

#WprimeZ1_M3000="/cms/mc/Samples/VBF_Wprime/Wprime_VBF_gWWpZ1_M3000.root"
#df_z1_M3000 = build_df(WprimeZ1_M3000)
#df_z1_M3000.to_csv("Wprime_VBF_gWWpZ1_M3000.csv")

#WprimeZ2_M500="/cms/mc/Samples/VBF_Wprime/Wprime_VBF_gWWpZ2_M500.root"
#df_z2_M500 = build_df(WprimeZ2_M500)
#df_z2_M500.to_csv("Wprime_VBF_gWWpZ2_M500.csv")

#WprimeZ2_M3000="/cms/mc/Samples/VBF_Wprime/Wprime_VBF_gWWpZ2_M3000.root"
#df_z2_M3000 = build_df(WprimeZ2_M3000)
#df_z2_M3000.to_csv("Wprime_VBF_gWWpZ2_M3000.csv")


def ManyIntoOne(Directory):
    samples = glob.glob(Directory)
    list_dfs = []
    for path in samples:
        list_dfs.append(build_df(path))
    
    df_total = pd.concat([dfi for dfi in list_dfs],ignore_index= True)
    print(f"Se produjo un csv con {df_total.shape[0]}")
    return df_total

#Wbkg="/cms/mc/Samples/VBF_Wprime/Wbkg/*.root"
#df_wbkg = ManyIntoOne(Wbkg)
#df_wbkg.to_csv("Wbkg_15032022.csv")

#TTbarbkg="/cms/mc/Samples/VBF_Wprime/TTbarbkg/*.root"
#df_ttbarbkg = ManyIntoOne(TTbarbkg)
#df_ttbarbkg.to_csv("TTbarbkg_15032022.csv")

WWbkg="/cms/mc/Samples/VBF_Wprime/WW_lvl_jj_bkg/*.root"
df_wwbkg = ManyIntoOne(WWbkg)
df_wwbkg.to_csv("../WWbkg_22032022.csv")

#WZ1bkg="/cms/mc/Samples/VBF_Wprime/WZ_lvl_jj_bkg/tag_1_delphes_events.root"
#df_wz1bkg = build_df(WZ1bkg)
#df_wz1bkg.to_csv("WZ1_15032022.csv")

#WZ2bkg="/cms/mc/Samples/VBF_Wprime/WZ2j_lvl_vv_bkg/tag_1_delphes_events.root"
#df_wz2bkg = build_df(WZ2bkg)
#df_wz2bkg.to_csv("WZ2_15032022.csv")
