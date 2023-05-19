
from cgi import print_arguments
from imghdr import what
from CLT_class import CLT
from MIXTURE_CLT import MIXTURE_CLT
from RANDOM_FOREST import RANDOM_FOREST
from Util import *

dsdir = "C:/Users/brett/source/repos/ML-Project4/ML-Project4/datasets/dataset/"

sets = ["accidents", "baudio", "bnetflix", "jester", "kdd", "msnbc", "nltcs", "plants", "pumsb_star", "tretail"]

'''
#-2.1-#
print("part 2.1:")
for s in sets:

    train = dsdir + s + ".ts.data"
    valid = dsdir + s + ".valid.data"
    test = dsdir + s + ".test.data"
    
    trainset = Util.load_dataset(train)
    validset = Util.load_dataset(valid)
    testset = Util.load_dataset(test)


    print(s + ": ")

    cltree = CLT()
    cltree.learn(trainset)

    vLL = cltree.computeLL(validset)/validset.shape[0]
    tLL = cltree.computeLL(testset)/testset.shape[0]

    print( "\tValidation set LL: " + str(vLL))
    print( "\tTest set LL: " + str(tLL))
'''
"""
#-2.2-#
print("part 2.2:")
for s in sets:
    train = dsdir + s + ".ts.data"
    valid = dsdir + s + ".valid.data"
    test = dsdir + s + ".test.data"

    trainset = Util.load_dataset(train)
    validset = Util.load_dataset(valid)
    testset = Util.load_dataset(test)

    components = [2, 5, 10, 20]

    for ncomponents in components:

        mix_tree = MIXTURE_CLT()

        mix_tree.learn(dataset=trainset, n_components=ncomponents)

        print(s + " with " + str(ncomponents) + " components: ")

        vLL = mix_tree.computeLL(validset)/validset.shape[0]
        tLL = mix_tree.computeLL(testset)/testset.shape[0]

        print( "\tValidation set LL: " + str(vLL))
        print( "\tTest set LL: " + str(tLL))
"""

"""
#-2.3-#
print("part 2.3:")
for s in sets:
    train = dsdir + s + ".ts.data"
    valid = dsdir + s + ".valid.data"
    test = dsdir + s + ".test.data"

    trainset = Util.load_dataset(train)
    validset = Util.load_dataset(valid)
    testset = Util.load_dataset(test)

    components = [2, 5, 10, 20]

    r_max = trainset.shape[1] * trainset.shape[1] / 2
    r = [r_max * 0.2, r_max * 0.4, r_max * 0.6, r_max * 0.8, r_max]

    for ncomponents in components:
      for r_ in r:
          mix_tree = RANDOM_FOREST()

          mix_tree.learn(dataset=trainset, n_components=ncomponents, r = int(r_))

          print(s + " with " + str(ncomponents) + " components and " + str(int(r_)) + " zero edges: ")

          vLL = mix_tree.computeLL(validset)/validset.shape[0]
          tLL = mix_tree.computeLL(testset)/testset.shape[0]

          print( "\tValidation set LL: " + str(vLL))
          print( "\tTest set LL: " + str(tLL))
"""

#-final-#
print("test results: ")
for s in sets:

    print(s)

    train = dsdir + s + ".ts.data"
    valid = dsdir + s + ".valid.data"
    test = dsdir + s + ".test.data"

    trainset = Util.load_dataset(train)
    validset = Util.load_dataset(valid)
    testset = Util.load_dataset(test)

    lls = []

    for i in range(5):
        model = None
        if s in ["accidents", "baudio", "bnetflix", "jester", "plants", "pumsb_star"]:
            model = MIXTURE_CLT()
            model.learn(dataset=trainset, n_components = 20)           
        if s in  ["kdd", "msnbc", "nltcs"]:
            model = MIXTURE_CLT()
            model.learn(dataset=trainset, n_components = 10)
        if s == "tretail":
            model = RANDOM_FOREST()
            model.learn(dataset=trainset, n_components = 250, r = 1822)

        lls.append(model.computeLL(testset)/testset.shape[0])

    avgll = np.average(lls)
    stddev = np.std(lls)

    print("\taverage: " + str(avgll))
    print("\tstddev: " + str(stddev))

