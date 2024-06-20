import os


def main():
  device = torch.device('cuda:0')
    model = Point_PN_PLM().to(device)
    model.eval()
    checkpoint = torch.load("./checkpoint/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    plm = torch.tensor(
        np.load('./data/negative.npy')).unsqueeze(0).to(device)
    # dim = 3(xyz) + 29 + 7(atom_types) + 16(dmasif)
    point_set = np.loadtxt('./data/negative.txt', delimiter=' ').astype(np.float32) 
    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
    point_set = torch.tensor(point_set).unsqueeze(0).transpose(2, 1).to(device)
    print(point_set.shape, plm.shape)
    out = model(point_set, plm)
    print(out)


if __name__ == '__main__':
    main()
