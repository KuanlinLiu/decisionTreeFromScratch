# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework 3 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.
from __future__ import print_function
import math
import numpy as np
from matplotlib import pyplot as plt

import os
import graphviz



def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE

    partn={}
    xdash=np.unique(x)
    for x1 in xdash:
        #print x1
        for i in range(len(x)):
            if(x[i]==x1):
                if x1 in partn:
                    temp=partn[x1]
                    temp.append(i)
                    partn[x1]=temp
                else:
                    temp=[]
                    temp.append(i)
                    partn[x1]=temp
    return partn

    raise Exception('Function not yet implemented!')


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))


    # INSERT YOUR CODE HERE"""
    py=0.0
    hy=0.0
    d=partition(y)
    for key in d:
        indices=d[key]
        temp=onesVector(indices,len(y))

        py=float(sum(temp))/float(len(y))

        hy+=(-1*py*math.log(py,2))
    return hy
    raise Exception('Function not yet implemented!')


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)


    # INSERT YOUR CODE HERE
    hy=entropy(y)
    #print hy
    hyx=0.0
    px=float(sum(x))/float(len(x))
    i=0
    d=partition(y)

    for valy in d:
        ind=onesVector(d[valy],len(y))
        py1=0.0
        for i in range(len(y)):
            if(ind[i] and x[i]):
                py1+=1.0

        pyx1=float(py1)/float(sum(x))
        if pyx1!=0.0:
            hyx+=(-1.0*pyx1*math.log(pyx1,2))
    #print hyx
    #print px

    #print(px * hyx)
    return hy-(px*hyx)"""
    dict_x = partition(x)
    #print(dict_x)
    #print(dict_x)

    entropy_y = entropy(y)

    #print("y", entropy_y)
    ans = 0
    # total_val=sum([len(x) for x in dict_x.values()])
    for key in dict_x:
        y_ans = []
        for i in dict_x[key]:
            y_ans.append(y[i])

        # print(y_ans)
        entropy_x = entropy(y_ans)
        #print(key)
        #print("entropy_x", entropy_x)
        prob = float(len(dict_x[key])) / float(len(x))

        #print(prob)
        ans += float(prob) * float(entropy_x)
    mi = entropy_y - ans
    #print("mi",mi)
    return mi


















    raise Exception('Function not yet implemented!')
def onesVector(indices,size):
    temp=[]
    for i in range(size):
        temp.append(0)
    for ix in indices:
        temp[ix]=1
    #print("temp:",temp)
    return temp

def getKey(avmi,maxmi):
    for key in avmi:
        if avmi[key]==maxmi:
            return key
def column(matrix, i):
    return [row[i] for row in matrix]

avpairs=[]

def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=3):
    tree={}


    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    if len(x)==0 or len(y)==0:
        return None

    if len(np.unique(y))==1:
        #print "tree"
        #print tree
        return y[0]

    if attribute_value_pairs!=None and len(attribute_value_pairs)==0:
        #print "here"
        ydash=set(y)
        maxcount=0
        maxy=0
        for y1 in ydash:
            count=0
            for i in range(len(y)):
                if y[i]==y1:
                    count+=1
            if count>maxcount:
                maxcount=count
                maxy=y1
        return maxy



    if depth==max_depth:
        #print "here too"
        ydash = set(y)
        maxcount = 0
        maxy=0
        for y1 in ydash:
            count = 0
            for i in range(len(y)):
                if y[i] == y1:
                    count += 1
            if count > maxcount:
                maxcount = count
                maxy=y1
        return maxy


    avmi={}
    maxmi=0
    maxcol=0
    maxattr=0
    ab=np.shape(x)
    #print (np.shape(x)[1])
   # xtemp=x.tolist()

    for i in range(ab[1]):

        d=partition(column(x,i))


        for key in d:

            #if (i,key) not in avpairs:
                indices=d[key]
                temp=onesVector(indices,len(x))
                mi=mutual_information(temp,y)
                #print(temp)
                avmi[(i,key)]=mi
    if attribute_value_pairs==None:
        attribute_value_pairs=avmi.keys()
    else:
        for key in avmi.keys():
            if key not in attribute_value_pairs:
                avmi.pop(key)

    if len(avmi)==0:
        return

    #print(avmi)

    maxmi=max(avmi.values())
    #print(maxmi)
    key=getKey(avmi,maxmi)

    (col, attr) = key
    #print col,attr
    #print key
    #avmi.pop(key)
    best_col_x=column(x,col)
    true_y=[]
    false_y=[]
    x_true=[]
    x_false=[]
    for i in range(len(y)):
        if best_col_x[i]==attr:
            true_y.append(y[i])
            x_true.append(x[i])
        else:
            false_y.append(y[i])
            x_false.append(x[i])


    cloned=attribute_value_pairs[:]
    cloned.remove(key)

    tree[(col,attr,True)]=id3(x_true,true_y,cloned,depth+1,max_depth)

    tree[(col, attr, False)]=id3(x_false,false_y,cloned,depth+1,max_depth)
#    print tree

    return tree



    raise Exception('Function not yet implemented!')

def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """
    key=tree.keys()

    (col,attr,val)=key[0]

    if x[col]==attr:
        #print("coll=attr")
        if type(tree[(col,attr,True)]) is dict:
            return predict_example(x,tree[(col,attr,True)])

        else:

            return tree[(col,attr,True)]

    else:
        #print("coll!=attr")
        if type(tree[(col,attr,False)]) is dict:
            return predict_example(x,tree[(col,attr,False)])

        else:
            return tree[(col,attr,False)]

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    #raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    #print(y_true)
    #print(y_pred)
    n=len(y_pred)
    count=0
    for i in range(n):
        if y_true[i]!=y_pred[i]:
            count+=1
    return float(count)/float(n)

    raise Exception('Function not yet implemented!')


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))



def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    #os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid
def geterrs(xtrn,ytrn,xtst,ytst,j):
    from sklearn.metrics import confusion_matrix
    confusions={}
    d={}
    for i in range(1,11):
        decision_tree=id3(xtrn,ytrn,max_depth=i)

        y_pred_trn = [predict_example(x, decision_tree) for x in xtrn]

        err_trn=compute_error(y_pred_trn,ytrn)
        y_pred_tst = [predict_example(x, decision_tree) for x in xtst]
        if i in [1, 3, 5]:
            confusions[i]=confusion_matrix(ytst,y_pred_tst)
            dot_str = to_graphviz(decision_tree)

            render_dot_file(dot_str, './treedepth'+str(i))
            import scikitplot as skp

            skp.metrics.plot_confusion_matrix(ytst,y_pred_tst)
            plt.title('figure of depth:'+str(i))
            plt.show()
        err_tst = compute_error(y_pred_tst, ytst)
        d[i]=(err_trn,err_tst)


    plot(d,j)
    return confusions



def plot(d,j):
    print(j)
    plt.figure()
    plt.title("figure"+str(j))
    trnerr=[]
    tsterr=[]
    depths=[]
    all=[]
    for i in range(1,11):
        (trn,tst)=d[i]
        trnerr.append(trn)
        tsterr.append(tst)
        depths.append(i)
        all.append(trn)
        all.append(tst)



    plt.plot(depths,trnerr, marker='o', linewidth=3, markersize=12)
    plt.plot(depths,tsterr, marker='s', linewidth=3, markersize=12)
    plt.ylabel('Train/Test error', fontsize=16)
    plt.xlabel('max_depth', fontsize=16)
    #plt.xticks(all, fontsize=12)

    plt.legend(['Train Error', 'Test Error'], fontsize=16)

    plt.show()




def skl(x,y,xtst,ytst):

    confusions={}
    for i in [1,3,5]:
        from sklearn import tree
        from sklearn.metrics import confusion_matrix, accuracy_score
        dtree = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=i)
        dtree = dtree.fit(x, y)
        #dot_str = tree.export_graphviz(dtree, out_file=None)
        #render_dot_file(dot_str, './my_learned_tree_sklearn of depth '+str(i))

        dtree = dtree.fit(x, y)
        dtree = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=i)
        dtree=dtree.fit(x,y)
        y_pred = dtree.predict(xtst)
        import scikitplot as skp
        skp.metrics.plot_confusion_matrix(ytst, y_pred)
        plt.title('figure of depth:' + str(i))
        plt.show()

        confusions[i]=confusion_matrix(ytst,y_pred)
        print(accuracy_score(ytst,y_pred))

    return confusions






if __name__ == '__main__':

    import pandas as pd

     # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]



    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    # Learn a decision tree of depth 3

    decision_tree = id3(Xtrn, ytrn, max_depth=3)


    # Pretty print it to console
    pretty_print(decision_tree,3)
    #confs=skl(Xtrn,ytrn,Xtst,ytst)
    #print("sklearn")
    #print(confs)
    #for depth in [1,3,5]:
        #ecision_tree = id3(Xtrn, ytrn, max_depth=depth)
    # Visualize the tree and save it as a PNG image
        #dot_str = to_graphviz(decision_tree)
        #render_dot_file(dot_str, './my_learned_tree_depth'+str(depth))

    # Compute the test error

    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)
    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
    #confs=geterrs(Xtrn,ytrn,Xtst,ytst,1)
    #print("monk1")
    #print(confs)
    #skl(Xtrn,ytrn,Xtst,ytst)


 
    '''
    M = np.genfromtxt('./monks-2.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    
    # Load the test data
    M = np.genfromtxt('./monks-2.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    confs = geterrs(Xtrn, ytrn, Xtst, ytst, 2)
    print("monk2")
    print(confs)

    M = np.genfromtxt('./monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    confs = geterrs(Xtrn, ytrn, Xtst, ytst, 2)
    print("monk3")
    print(confs)
    
   





    M=pd.read_csv('car_data.data')

    M = M.replace('vhigh', 4)
    M = M.replace('high', 3)
    M = M.replace('med', 2)
    M = M.replace('low', 1)
    M = M.replace('5more', 6)
    M = M.replace('more', 5)
    M = M.replace('vhigh', 4)
    M = M.replace('small', 1)
    M = M.replace('med', 2)
    M = M.replace('big', 3)
    M = M.replace('med', 2)
    M = M.replace('good', 3)
    M = M.replace('vgood', 4)
    M = M.replace('unacc', 1)
    M = M.replace('acc', 2)

    from sklearn.model_selection import train_test_split
    X=M.iloc[:,:-1].values

    y=M.iloc[:,-1].values
    Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2)
    #print(set(ytrn))
    #confs = geterrs(Xtrn, ytrn, Xtst, ytst, 1)
    #skl(Xtrn, ytrn, Xtst, ytst)


    M= pd.read_csv('hayes_roth.data',delimiter=',',header=None,dtype=int)
    test=pd.read_csv('hayes_roth.test',delimiter=',',header=None,dtype=int)



    from sklearn.model_selection import train_test_split

    Xtrn = M.iloc[:, 1:-1].values

    ytrn = M.iloc[:, -1].values

    Xtst=test.iloc[:,:-1].values
    ytst=test.iloc[:,-1].values

    #confs = geterrs(Xtrn, ytrn, Xtst, ytst, 1)
    skl(Xtrn, ytrn, Xtst, ytst)
'''
