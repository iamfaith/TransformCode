try:
    from base_operator import BaseOperator
except:
    from .base_operator import BaseOperator
import random
import math
import copy


class WhileForExchange(BaseOperator):

    def __init__(self, language: str):
        super(WhileForExchange, self).__init__(language)

    def whileForExchange(self, code_snippet):

        tree = self.parse(code_snippet)
        statements = self.get_nodes(code_snippet, tree.root_node,
                                    debug=False, capture_types=["for_statement", "while_statement"], loopChildren=True)
        # print(statements)
        # oriContent = code_snippet.encode()
        content = code_snippet.encode()
        if len(statements) > 0:
            for_ignore_node_kind = ["for", "("]

            previous = None

            for idx, (statement, token) in enumerate(reversed(statements)):
                for2while = True if self.getNodeType(statement.children[0]) == "for" else False
                if idx > 0:
                    _tree = self.parse(content.decode())
                    _statements = self.get_nodes(content.decode(), _tree.root_node, debug=False, capture_types=["for_statement", "while_statement"], loopChildren=True)
                    statement, _ = list(reversed(_statements))[idx]
                    
                start, end = statement.start_byte, statement.end_byte
                
                remain_statements, startCollect = [], False
                updateOperation = []
                if for2while:
                    for node in statement.children:        
                        nodeType = self.getNodeType(node)
                        previous = node
                        if nodeType in for_ignore_node_kind:
                            continue
                        else:
                            if not startCollect and nodeType != ')':
                                updateOperation.append(node)
                        if nodeType == ")":
                            startCollect = True
                            continue
                        if startCollect:
                            remain_statements.append(node)
                else:
                    nodes = statement.children
                    conditionNodes, blockNodes = nodes[1], nodes[2]
                    conditionToken = conditionNodes.text.decode()
                    conditionToken = conditionToken.replace("(", "").replace(")", "")

                    for_token = 'for(;{};)'.format(conditionToken).encode()
                    remain_token = blockNodes.text
                    content = content[:start] + for_token + remain_token + content[end:]
                    
                    

                if for2while:
                    # update for statement
                    for_token = ''
                    for op in updateOperation:
                        for_token += op.text.decode()

                    try:
                        defineOp, conditionOp, updateOp = for_token.split(';')
    
                        remain_token = ''
                        for op in remain_statements:
                            remain_token = op.text.decode()

                        if updateOp != "":
                            updateOp = updateOp.split(',')
                            updateOp = ';'.join(updateOp)
                            if remain_token[-1] == "}":
                                remain_token = remain_token[:-1] + updateOp + ";" + "}"
                            else:
                                remain_token = remain_token[:-1] + updateOp + ";"

                        new_statement = '{};while({}){}'.format(defineOp, conditionOp, remain_token)
                        content = content[:start] + new_statement.encode() + content[end:]
                    except Exception as e:
                        # print("[for2while] error", for_token)
                        pass

        return statements, content.decode('utf-8', 'ignore')


def main():
    whileForExchange = WhileForExchange(language="c")

    sample_codes = [
        # """
        # void bubbleSort(int arr[], int n)
        # {
         
        #        // Last i elements are already in place  
        #        for (int j = 0; j < i; j++) {
        #            if (arr[j] > arr[j+1])
        #               swap(&arr[j], &arr[j+1]);
        #               swap(a, b, 10)
        #         }   
        #         int n = 100;
        #         while(n > 0 )
        #         {
        #             n --;
        #         }  
        #         for (int i = 0; i < 100; i++) {

        #         }
        # }
        # """,
        """
int _Partition(int *A, int left, int right)
{
    int pivot = _Pivot(A, left, right);
    int i = left + 1, j = right - 2;
    while (1)
    {
        while (A[i] < pivot)
            i++;
        while (A[j] > pivot)
            j--;
        if (i < j)
        {
            swap(A, i, j);
            i++;
            j--;
        }
        else
            break;
    }
    swap(A, i, right - 1);
    return i;
}"""
    ]

    for sample_code in sample_codes:
        # swap(sample_code)
        permute_codes, code = whileForExchange(sample_code)
        print("-" * 50)
        for i in permute_codes:
            print(i)
        print("-" * 50)
        print(code)
        break


if __name__ == "__main__":
    main()
