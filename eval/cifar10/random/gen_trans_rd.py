from genotypes import PRIMITIVES
from genotypes import Genotype_op
import numpy as np
import sys
import argparse
import pickle as pk
import argparse
import math

np.set_printoptions(suppress=True)
def genotype_unbind(alphas,args):

    def _distribution_quantile_index(distribution,threshold,threshold_element_wise=0):
        sorted_index=np.argsort(-distribution)
        sorted_value=np.sort(distribution)[::-1]
        distribution_cumsum=sorted_value.cumsum()
        if threshold_element_wise>0: quantile_index=(sorted_value>=threshold_element_wise).sum()
        elif threshold>1: quantile_index=math.ceil(threshold)
        else: quantile_index=np.argmax(distribution_cumsum>threshold)+1
        return sorted_index,quantile_index

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(args.steps):
        end = start + n
        #W_o = np.around(weights[start:end].copy(),decimals=4)
        W = weights[start:end].ravel()
        W_sorted_index,quantile_index=_distribution_quantile_index(W,args.operations_cumsum_threshold)
        for a in range(quantile_index):
            op_index=W_sorted_index[a]%len(PRIMITIVES)
            pre_node_index=W_sorted_index[a]//len(PRIMITIVES)
            gene.append((PRIMITIVES[op_index],pre_node_index,i))
        start = end
        n += 1
      return gene

    #math.isclose(0.1 + 0.2, 0.3)
    gene_normal = _parse(alphas[0])
    gene_reduce = _parse(alphas[1])
    #alpha_normal_output = alphas[2]
    #alpha_reduce_output = alphas[3]
    while True:
        alpha_normal_output=np.random.randint(0, 2, size=alphas[2].shape)
        alpha_reduce_output=np.random.randint(0, 2, size=alphas[3].shape)
        #print(alpha_normal_output)
        #print(alpha_reduce_output)
        if alpha_normal_output[0].any() and alpha_normal_output[1].any() and alpha_reduce_output[0].any() and alpha_reduce_output[1].any(): break
    gene_normal_output={}
    gene_reduce_output={}

    normal_concat_path_wise_threshold=0; normal_concat_cumsum_threshold=0
    reduce_concat_path_wise_threshold=0; reduce_concat_cumsum_threshold=0
    if args.normal_concat_path_wise_threshold!=0:
        if args.normal_concat_path_wise_threshold<1:
            normal_concat_path_wise_threshold=args.normal_concat_path_wise_threshold
        else: normal_concat_cumsum_threshold=args.normal_concat_path_wise_threshold
    else:
        normal_concat_path_wise_threshold=alpha_normal_output[1].mean()

    if args.reduce_concat_path_wise_threshold!=0:
        if args.reduce_concat_path_wise_threshold<1:
            reduce_concat_path_wise_threshold=args.reduce_concat_path_wise_threshold
        else: reduce_concat_cumsum_threshold=args.reduce_concat_path_wise_threshold
    else:
        reduce_concat_path_wise_threshold=alpha_reduce_output[1].mean()

    normal_output_sorted_index,quantile_index=_distribution_quantile_index(alpha_normal_output[0],args.sum_output_cumsum_threshold,args.sum_path_wise_threshold)
    gene_normal_output['sum']=normal_output_sorted_index[:quantile_index].tolist()
    normal_output_sorted_index,quantile_index=_distribution_quantile_index(alpha_normal_output[1],normal_concat_cumsum_threshold,normal_concat_path_wise_threshold)
    gene_normal_output['concat']=normal_output_sorted_index[:quantile_index].tolist()

    reduce_output_sorted_index,quantile_index=_distribution_quantile_index(alpha_reduce_output[0],args.sum_output_cumsum_threshold,args.sum_path_wise_threshold)
    gene_reduce_output['sum']=reduce_output_sorted_index[:quantile_index].tolist()
    reduce_output_sorted_index,quantile_index=_distribution_quantile_index(alpha_reduce_output[1],reduce_concat_cumsum_threshold,reduce_concat_path_wise_threshold)
    gene_reduce_output['concat']=reduce_output_sorted_index[:quantile_index].tolist()

    gene_normal_filtered=[]
    for index,node in enumerate(gene_normal):
        _,input_index,node_index=node
        if node_index in gene_normal_output['concat']+gene_normal_output['sum']: 
            gene_normal_filtered.append(gene_normal[index])
        else:
            intermediate_nodes=[node_index]
            for a in range(index+1,len(gene_normal)):
                if gene_normal[a][1]-2 in intermediate_nodes:
                    intermediate_nodes.append(gene_normal[a][2])
                    if intermediate_nodes[-1] in gene_normal_output['concat']+gene_normal_output['sum']: 
                        gene_normal_filtered.append(gene_normal[index]) 
                        break

    gene_reduce_filtered=[]
    #print(gene_reduce)
    for index,node in enumerate(gene_reduce):
        _,input_index,node_index=node
        if node_index in gene_reduce_output['concat']+gene_reduce_output['sum']: 
            gene_reduce_filtered.append(gene_reduce[index])
        else:
            intermediate_nodes=[node_index]
            for a in range(index+1,len(gene_reduce)):
                if gene_reduce[a][1]-2 in intermediate_nodes:
                    intermediate_nodes.append(gene_reduce[a][2])
                    if intermediate_nodes[-1] in gene_reduce_output['concat']+gene_reduce_output['sum']: 
                        gene_reduce_filtered.append(gene_reduce[index]) 
                        break

    #concat = range(2+steps-4, steps+2)
    genotype = Genotype_op(
      normal=gene_normal_filtered,
      normal_output=gene_normal_output,
      reduce=gene_reduce_filtered,
      reduce_output=gene_reduce_output,
    )
    return genotype

if __name__ == '__main__':
    parser = argparse.ArgumentParser("cifar")
    test=[]
    parser.add_argument('--arch_file', type=str, help='location of the data corpus')
    parser.add_argument('--operations_cumsum_threshold', type=float, default='2',help='location of the data corpus')
    parser.add_argument('--sum_output_cumsum_threshold', type=float, default='0.9',help='location of the data corpus')
    parser.add_argument('--steps', type=int, default='5',help='location of the data corpus')
    parser.add_argument('--sum_path_wise_threshold', type=float, default='0.2',help='location of the data corpus')
    parser.add_argument('--normal_concat_path_wise_threshold', type=float, default='0',help='location of the data corpus')
    parser.add_argument('--reduce_concat_path_wise_threshold', type=float, default='0',help='location of the data corpus')
    args = parser.parse_args()
    alphaList=pk.load(open(args.arch_file,'rb'))
    print(alphaList)
    print(genotype_unbind(alphaList,args))
