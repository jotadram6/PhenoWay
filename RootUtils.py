import ROOT
from array import array

def NpRootConverter(FileName, HistoName, ListBins, ListContent, ListErrors):
  """Usage:
  Bins=[10,20,30,40,50,60] # The last value is the maximum of the last histogram bin
  Content=[1,2,3.1,4.1,3.1,2.1] #Size must be len(bins)-1
  Errors=[0.1,0.2,0.3,0.4,0.3,0.2] #Size must be len(bins)-1

  NpRootConverter("ExampleFile", "histoexample", Bins, Content, Errors)
  """
    MyFile = ROOT.TFile(FileName+".root","recreate")
    MyH = ROOT.TH1F(HistoName, HistoName, len(ListBins)-1, array('d',ListBins))
    for i in range(len(ListContent)):
        MyH.SetBinContent(i+1,ListContent[i])
        MyH.SetBinError(i+1,ListErrors[i])
    MyFile.Write()
