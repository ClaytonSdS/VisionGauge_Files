import os
from sklearn.preprocessing import StandardScaler
import pandas as pd 

def Folder2DataFrame(path):

    # pega só as pastas do primeiro nível (ex: ['train', 'valid'])
    folders = [
        item for item in os.listdir(path)
        if os.path.isdir(os.path.join(path, item))
    ]

    df = pd.DataFrame(columns=['file', 'deltaH_cm'])

    # percorre cada pasta de primeiro nível
    for fold in folders:
        fold_path = os.path.join(path, fold)

        # agora desce dentro dessa pasta e pega os arquivos
        for raiz, pastas, arquivos in os.walk(fold_path):

            # se tiver arquivos, eles são as imagens
            for arquivo in arquivos:
                caminho_imagem = os.path.join(raiz, arquivo)

                # pega o rótulo a partir do nome da pasta
                label = os.path.basename(raiz)

                df.loc[len(df)] = [caminho_imagem, float(label)]

    return df



class SETTINGS:
    def __init__(self):
        self.USE_SCALER = False

        my_current_dir = os.getcwd()

        self.DATASET_DIR = os.path.join(my_current_dir, "dataset", "training", "UTM_Dataset.v2-1k_nosplit.folder", "train")
  
        self.LOCALIZER = Folder2DataFrame(self.DATASET_DIR)
        self.LOCALIZER["deltaH_mm"] = self.LOCALIZER["deltaH_cm"] * 10 # convertendo para mm
    
        # SCALER CONFIG
        self.SCALER = StandardScaler()
        self.SCALER.fit(self.LOCALIZER["deltaH_cm"].values.reshape(-1, 1))
        
        if self.USE_SCALER:
            self.LOCALIZER["DELTA_SCALED"] = self.SCALER.transform(self.LOCALIZER["deltaH_cm"].values.reshape(-1, 1))


    
        VERBOSE = False
       
        # TRAINING
        self.TRAINING = True
        self.BATCH_SIZE = 64
        self.K_FOLD = 2
        self.NUM_EPOCHS = 30
        self.LR = 1e-5
        self.MAX_CACHE = 100
        self.PATIENCE = 10
    