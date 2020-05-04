import numpy as np
from numpy import array_equal as ae


def cm2(predicted, actual):
    p = np.array(predicted)
    a = np.array(actual)

    tn = 0  # True Normal
    tp = 0  # True Pneumonia
    fn = 0  # False Normal
    fp = 0  # False Pneumonia

    wrong = 0
    wr = []

    norm = np.array([1, 0])
    pneu = np.array([0, 1])

    for i in range(p.shape[0]):
        pred = p[i]
        actu = a[i]

        if ae(pred, actu):
            if ae(pred, norm):
                tn += 1
            elif ae(pred, pneu):
                tp += 1
            else:
                wrong += 1
                wr.append(pred)
        else:
            if ae(pred, norm):
                fn += 1
            elif ae(pred, pneu):
                fp += 1
            else:
                wrong += 1
                wr.append(pred)

    tan = tn + fn       # Total Actual Normal
    tap = tp + fp       # Total Actual Pneumonia
    tpn = tn + fp       # Total Predicted Normal
    tpp = fn + tp       # Total Predicted Pneumonia

    print("{0} \t\t {1} \t\t {2}".format(" ", norm, pneu))
    print("{0} \t {1:5d} \t\t {2:5d} \t\t{3:5d}".format(norm, tn, fn, tan))
    print("{0} \t {1:5d} \t\t {2:5d} \t\t{3:5d}".format(pneu, fp, tp, tap))
    print("{0} \t\t {1:5d} \t\t {2:5d} \t\t{3:5d}".format(" ", tpn, tpp, (tan + tap)))

    print("Wrong:", wrong)
    print(wr)


def cm3(predicted, actual):
    p = np.array(predicted)
    a = np.array(actual)

    tn = 0  # True Normal
    tb = 0  # True Bacterial
    tv = 0  # True Viral

    fbn = 0  # Pred Bacterial, Act Normal
    fbv = 0  # Pred Bacterial, Act Viral
    fvn = 0  # Pred Viral, Act Normal
    fvb = 0  # Pred Viral, Act Bacterial
    fnb = 0  # Pred Normal, Act Bacterial
    fnv = 0  # Pred Normal, Act Viral

    wrong = 0

    norm = np.array([1, 0, 0])
    bact = np.array([0, 1, 0])
    vira = np.array([0, 0, 1])

    wr = []

    for i in range(p.shape[0]):
        pred = p[i]
        actu = a[i]

        if ae(pred, norm):
            if ae(actu, norm):
                tn += 1
            elif ae(actu, bact):
                fnb += 1
            elif ae(actu, vira):
                fnv += 1
            else:
                wrong += 1
                wr.append(pred)
        elif ae(pred, bact):
            if ae(actu, norm):
                fbn += 1
            elif ae(actu, bact):
                tb += 1
            elif ae(actu, vira):
                fbv += 1
            else:
                wrong += 1
                wr.append(pred)
        elif ae(pred, vira):
            if ae(actu, norm):
                fvn += 1
            elif ae(actu, bact):
                fvb += 1
            elif ae(actu, vira):
                tv += 1
            else:
                wrong += 1
                wr.append(pred)
        else:
            wrong += 1
            wr.append(pred)

    t = predicted.shape[0]
    print("\n============ Confusion Matrix =============")
    print("{0} \t\t\t {1} \t {2} \t{3}".format(" ", norm, bact, vira))
    print("{0} \t {1:5d} \t\t {2:5d} \t\t {3:5d} \t {4:5d}".format(norm, tn, fbn, fvn, (tn + fbn + fvn)))
    print("{0} \t {1:5d} \t\t {2:5d} \t\t {3:5d} \t {4:5d}".format(bact, fnb, tb, fvb, (fnb + tb + fvb)))
    print("{0} \t {1:5d} \t\t {2:5d} \t\t {3:5d} \t {4:5d}".format(vira, fnv, fbv, tv, (fnv + fbv + tv)))
    print("{0} \t\t\t {1:5d} \t\t {2:5d} \t\t {3:5d} \t {4:5d}".format(" ", (tn + fnb + fnv), (fbn + tb + fbv),
                                                                       (fvn + fvb + tv), t))
    print("Not Counted:", wrong)
    print(wr)
