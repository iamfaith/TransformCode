try:
    from base_operator import BaseOperator
except:
    from .base_operator import BaseOperator
import random
import math
import copy


class ArithmeticTransform(BaseOperator):

    def __init__(self, language: str):
        super(ArithmeticTransform, self).__init__(language)



    def arithmeticTransform(self, code_snippet):

        tree = self.parse(code_snippet)
        statements = self.get_nodes(code_snippet, tree.root_node, debug=False, capture_types=["update_expression", "assignment_expression"])
        content = code_snippet.encode()
        oriContent = code_snippet.encode()
        statements.sort(key=lambda x: x[0].start_byte, reverse=False)
        for node, token in reversed(statements):
            start, end = node.start_byte, node.end_byte
            if self.getNodeType(node) == "update_expression":
                
                if self.getNodeType(node.children[0]) == 'identifier':
                    identifyNode, op =  node.children[0], node.children[1]
                else:
                    op, identifyNode =  node.children[0], node.children[1]
                opText = op.text[:1]
                if random.random() > 0.5:
                    # a = a + 1
                    newToken = identifyNode.text + "=".encode() + identifyNode.text + opText + "1".encode()
                else:
                    # a += 1
                    newToken = identifyNode.text +  opText + "=".encode() + "1".encode()
                content = content[:start] + newToken + content[end:]
            else:
                if self.getNodeType(node.children[0]) == 'identifier':
                    identifyNode, op =  node.children[0], node.children[1]
                    rightPart = node.children[2]
                else:
                    identifyNode, op =  node.children[2], node.children[1]
                    rightPart = node.children[0]
                opText = op.text[:1]
                if self.getNodeType(op) in ["+=", "-=", "*=", "\=", "**=", "^="]:
                    # a += 1 -> a = a + 1
                    newToken = identifyNode.text + "=".encode() + identifyNode.text + opText + rightPart.text
                    content = content[:start] + newToken + content[end:]
                else:
                    if self.getNodeType(rightPart) == 'binary_expression':
                        rightPartIdentify = rightPart.children[0]
                        if self.getNodeType(rightPartIdentify) == "identifier" and rightPartIdentify.text == identifyNode.text:
                            # i = i *3 -> i *= 3
                            newToken = identifyNode.text + rightPart.children[1].text + "=".encode() + rightPart.children[2].text
                            content = content[:start] + newToken + content[end:]
                        elif self.getNodeType(rightPartIdentify) == "binary_expression":
                            parentNode = self._loop(rightPartIdentify)
                            _identify = parentNode.children[0]
                            if self.getNodeType(_identify) == "identifier" and _identify.text == identifyNode.text:
                                # a = a + 1 *9 + 20 -> a += 1 *9 + 20
                                start_pos = parentNode.children[1].end_byte
                                newToken = identifyNode.text + parentNode.children[1].text + "=".encode() + content[start_pos:end]
                                content = content[:start] + newToken + content[end:]
                    elif self.getNodeType(rightPart) == 'array_access':
                        try:
                            exp_nodes = self.get_nodes(code_snippet, rightPart, debug=False, ignore_types=["comment"], capture_types=["binary_expression"])
                            if len(exp_nodes) > 0:
                                exp_node = exp_nodes[0][0]
                                if exp_node.child_count == 3:
                                    start, end = exp_node.start_byte, exp_node.end_byte
                                    newToken = exp_node.children[2].text + exp_node.children[1].text + exp_node.children[0].text
                                    content = content[:start] + newToken + content[end:]
                        except:
                            print('----')
                            
                          
                    # else:
                        # print("--", token)

        return statements, content.decode('utf-8', 'ignore')

    def _loop(self, node):
        if node.child_count > 0 and node.children[0].child_count == 0:
            return node
        return self._loop(node.children[0])



def main():
    arithmetic = ArithmeticTransform(language="java")

    sample_codes = [
        """
        void bubbleSort(int arr[], int n)
        {
           int i = 0;
           i += 2 + 10;
           i = i +3;
           int j = 0;
           i = 1000;
           call(i, j);
           int k, b;
           i = k *b;
        int a = 0;
    a = a + 1 *9 + 20 * a;
        a = a + 20
           for (i = 0; i < n-1; i--)     
           {
                j++;
           }
        }
        """
    ]

    for sample_code in sample_codes:
        permute_codes, code = arithmetic(sample_code)
        print("-" * 50)
        for i in permute_codes:
            print(i)
        print("-" * 50)
        print(code)
        break


if __name__ == "__main__":
    main()
