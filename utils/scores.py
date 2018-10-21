import numpy as np

def dice_score(volumes_s,volumes_t,nclasses):
    assert(len(volumes_s) == len(volumes_t))
    n_ = len(volumes_s)
    scores = np.zeros((n_,nclasses),dtype=np.float32)
    for idx in range(len(volumes_s)):
        vol1 = volumes_s[idx]
        vol2 = volumes_t[idx]

        for c in range(nclasses):
            mask1 = (vol1 == c).astype(np.uint8)
            mask2 = (vol2 == c).astype(np.uint8)

            d = 2*(np.sum(np.logical_and(mask1 == 1, mask2 == 1).astype(np.uint8)))/float(np.sum((mask1==1).astype(np.uint8)) + np.sum((mask2==1).astype(np.uint8)))

            scores[idx,c] = d
    return scores

def dice_score_3d(volumes_s,volumes_t,nclasses):
    assert(volumes_s.shape == volumes_t.shape)
    scores = np.zeros((volumes_s.shape[0],nclasses))
    for i in range(volumes_s.shape[0]):
        vol1 = volumes_s[i]
        vol2 = volumes_t[i]

        for c in range(nclasses):
            mask1 = (vol1 == c).astype(np.uint8)
            mask2 = (vol2 == c).astype(np.uint8)
            d = 2*(np.sum(np.logical_and(mask1 == 1, mask2 == 1).astype(np.uint8)))/float(np.sum((mask1==1).astype(np.uint8)) + np.sum((mask2==1).astype(np.uint8)))

            scores[i,c] = d
    return scores
