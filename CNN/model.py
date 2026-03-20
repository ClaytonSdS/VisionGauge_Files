import torch
import torch.nn as nn
from torchvision import models

class NewDirectModel(nn.Module):
    def __init__(self, backbone_name: str, image_size:tuple=(120, 120), unfreeze_all:bool=False, use_head:bool = False, debug: bool = False):
        super().__init__()
        self.model_name = backbone_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug
        self.use_head = use_head

        self.unfreeze_all = unfreeze_all # descongelar tudo e manter o shape da arquitetura

        # placeholder image_size; será definido em load_backbone
        self.image_size = image_size

        # Carrega o backbone e constrói a head
        self.load_backbone()

        self.to(self.device)

    def build_head(self, output_features):
        self.head = nn.Sequential(
                    nn.Linear(output_features, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(512 , 1),
                )

        for p in self.head.parameters():
                p.requires_grad = True

        print("Descongelado os parâmetros de head")

        parameters = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        print(f"Head MLP Parâmetros Treináveis: {parameters}")


    def forward(self, x):
        #x = x.to(self.device)

        #print("FORWARD INPUT DEVICE:", x.device)
        #print("BACKBONE DEVICE:", next(self.backbone.parameters()).device)

        # Forward: Backbone -> Head -> Output
        if self.use_head:
            features = self.backbone(x)
            return self.head(features)


        # Forward: Backbone -> Output
        else:
            return self.backbone(x)


    def load_backbone(self):
        name = self.model_name.lower()

        # RESNET =======================================================================================================================================
        if name in ("resnet", "resnet18"):
            self.image_size = (120, 120)
            m = models.resnet18(pretrained=True)

            out_feats = m.fc.in_features

            # Usar head como MLP para regressão
            if self.use_head:
                m.fc = nn.Identity() # remover a última FC
                self.build_head(output_features=out_feats) # criar head mlp

            # Sem head: usa FC de saída única -> (1280, 1)
            else:
                m.fc = nn.Linear(out_feats, 1)

            # Congelar tudo
            for p in m.parameters():
                p.requires_grad = False

            # Descongelar tudo se  unfreeze_all == True
            if self.unfreeze_all:
                for p in m.parameters():
                    p.requires_grad = True

            # Descongelar apenas o classificador (FC)
            else:
                for param in m.layer3.parameters():
                  param.requires_grad = True

                print("Resnet Features [3] Descongelado")

                for param in m.layer4.parameters():
                  param.requires_grad = True

                print("Resnet Features [4] Descongelado")

                for p in m.fc.parameters():
                    p.requires_grad = True

            self.backbone = m
