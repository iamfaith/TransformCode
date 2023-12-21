try:
    from base_operator import BaseOperator
except:
    from .base_operator import BaseOperator
import random
import math
import copy


class SwapCondition(BaseOperator):

    def __init__(self, language: str):
        super(SwapCondition, self).__init__(language)
        # no ||
        self.canSwapOperation = set(["!=", "==", ">", "<", ">=", "<=", "&", "&&", "+"])
        self.swapDict = {
            ">": "<",
            "<": ">",
            ">=": "<=",
            "<=": ">=",
            "+": "+",
            "-": "-",
            "*": "*",
            "&&": "&&",
            "&": "&",
            "==": "==",
            "!=": "!="
        }
        # self.lowerPrio = set(["!=", "==", ">", "<", ">=", "<=", "&", "&&", "+",  "*"])

    # def _swap(self, node, content):
    #     # print(len(node.children), token, self.getTokenByte(node.children[0], oriContent), self.getTokenByte(node.children[-1], oriContent))
    #     node0 = node.children[0]
    #     node2 = node.children[2]

    #     op = node.children[1]
    #     op_text = self.getTokenByte(op, content).decode()
    #     if op_text in self.swapDict.keys():
    #         # print("--")
    #         op_text = self.swapDict.get(op_text)
    #         # pass

    #     c = self.getTokenByte(node, content)
    #     ret = node2.text + op_text.encode() + node0.text
    #     gap = len(c) - len(ret)
    #     if gap > 0:
    #         ret += b' ' * gap
    #     return ret

    def _swap(self, node, content):
        # print(len(node.children), token, self.getTokenByte(node.children[0], oriContent), self.getTokenByte(node.children[-1], oriContent))

        node0 = node.children[0]
        op = node.children[1]
        node2 = node.children[2]
        c = self.getTokenByte(node, content)
        if op.text.decode() in ["+", "-"]:
            loopNode = None
            if self.getNodeType(node0) == "binary_expression":
                loopNode = node0

            if self.getNodeType(node2) == "binary_expression":
                loopNode = node2
            if loopNode:
                ops = []

                def loop(node, ops):
                    if node.child_count == 0:
                        return node.text.decode()
                    for n in node.children:
                        ret = loop(n, ops)
                        if ret:
                            ops.append(ret)
                loop(loopNode, ops)
                ops = set(ops) & self.canSwapOperation
                # print(ops)
                if len(ops) == 1:
                    op = list(ops)[0]

                    swapOps = c.decode().split(op)
                    first, last = swapOps[0], swapOps[-1]
                    return ''.join([last, self.swapDict[op], first]).encode()

        node0_start_byte = node.children[0].start_byte - node.start_byte
        node0_end_byte = node.children[0].end_byte - node.start_byte

        node2_start_byte = node.children[2].start_byte - node.start_byte
        node2_end_byte = node.children[2].end_byte - node.start_byte

        op_start_byte = node.children[1].start_byte - node.start_byte
        op_end_byte = node.children[1].end_byte - node.start_byte

        op_text = self.getTokenByte(op, content).decode()
        if op_text in self.swapDict.keys():
            # print("--")
            op_text = self.swapDict.get(op_text)
            # pass
        # print("!!", node2.text, c[node2_start_byte:node2_end_byte])
        # print("!!", node0.text, c[node0_start_byte:node0_end_byte])
        # print("!!", op.text, c[op_start_byte:op_end_byte])

        new_node0 = c[node2_start_byte:node2_end_byte]
        new_node2 = c[node0_start_byte:node0_end_byte]

        if op_text == "==":
            if new_node0.decode() in ["true", "false"]:
                return ("false" if new_node0.decode() == "true" else "true").encode() + c[node0_end_byte:op_start_byte] + "!=".encode() + c[op_end_byte:node2_start_byte] + new_node2
            elif new_node2.decode() in ["true", "false"]:
                return new_node0 + c[node0_end_byte:op_start_byte] + "!=".encode() + c[op_end_byte:node2_start_byte] + ("false" if new_node2.decode() == "true" else "true").encode()

        ret = new_node0 + c[node0_end_byte:op_start_byte] + \
            op_text.encode() + c[op_end_byte:node2_start_byte] + new_node2

        # new_ret =
        return ret

    def canSwap(self, node, content, addition=None):
        token = self.getChildType(node, idx=1)
        if token in self.canSwapOperation:
            return True

        if addition is not None and token in addition:
            return True
        return False
        # token = self.getTokenByte(node, content).decode()
        # for op in self.canSwapOperation:
        #     if op in token:
        #         return True
        # return False

    def checkChild(self, child, binary_ops, content):
        if child.child_count == 0:
            return
        for node in reversed(child.children):
            if node is not None:
                self.checkChild(node, binary_ops, content)
                if self.getNodeType(node) in ['binary_expression']:
                    # if node.child_count == 3 and self.getChildType(node, idx=0) == "identifier" and self.getChildType(node, idx=2) == "identifier":
                    if self.canSwap(node, content, addition=["-"]):
                        for child in node.children:
                            if self.getNodeType(child) in ["&&", "||"]:
                                return

                        token = self.getTokenByte(node, content)
                        delOp = None
                        for op in binary_ops:
                            op_token = self.getTokenByte(op, content)
                            if op_token in token:
                                delOp = op
                                break
                        if delOp is not None:
                            binary_ops.remove(delOp)
                        binary_ops.append(node)

        # i = 100
        # j = 200
        # if i < j -1:
        #     pass

    def swapCondition(self, code_snippet):
        # print(self.__dict__)
        tree = self.parse(code_snippet)
        statements = self.get_nodes(code_snippet, tree.root_node,
                                    debug=False, capture_types=["binary_expression"])
        oriContent = code_snippet.encode()
        content = code_snippet.encode()

        statements = sorted(statements, key=lambda x: x[0].start_byte)
        for node, token in reversed(statements):
            binary_ops = []
            self.checkChild(node, binary_ops, oriContent)
            if self.canSwap(node, oriContent, addition=['-']):
                delOp = []
                for op in binary_ops:
                    # need to del
                    if op.parent.children[1].text.decode() in ["+", "-"]:
                        delOp.append(op)
                    # op_token = self.getTokenByte(op, content).decode()
                    # if op_token in token:
                    #
                    #     break
                if len(delOp) > 0:
                    for o in delOp:
                        binary_ops.remove(o)
                if len(binary_ops) == 0:
                    binary_ops.append(node)
                if self.canSwap(node, oriContent, addition=['-']) and node not in binary_ops:
                    binary_ops.insert(0, node)

            # print('-' * 10)
            # for op in binary_ops:
            #     print(self.getTokenByte(op, oriContent).decode(), "|" + token)
            # print('-' * 10)
            # tokenByte = self._swap(node, token, oriContent).encode()

            for i, op in enumerate(reversed(binary_ops)):
                c = self._swap(op, oriContent)
                c_origin = self.getTokenByte(op, oriContent)
                start, end = op.start_byte, op.end_byte
                # content = content[:start] + c + content[end:]
                content = content[:start] + c + content[end:]
                # print('-----!', c.decode())
            # print('---', len(binary_ops))

            # print(i, start, end, tokenByte.decode())
            # print(content.decode('utf-8', 'ignore'))
        try:
            return statements, content.decode('utf-8', 'ignore')
        except Exception as e:
            return statements, code_snippet


    def swapConditionClass(self, code_snippet):
        return self.swapCondition(code_snippet)


def main():
    swap = SwapCondition(language="java")

    sample_codes = [
        """
        public class Main {
          public static void main(String[] args) {
            int a = 15;
            Test.func1(a);
            circle.circumference(100, 20);
            if (b > 12) {
                label123: println("continuing");
                System.out.println("hello");
                continue;
                break;
                ;
                return 1;
            }
          }
        }
        """,
        """
        void bubbleSort(int arr[], int n)
        {
         
               // Last i elements are already in place  
               for (j = 0; j < n-i-1; j++)
                   if (arr[j] > arr[j+1])
                      swap(&arr[j], &arr[j+1]);
                      swap(a, b, 10);
                if (a >= b + 2 && a < c * 3  && a > b + 100 || d > 20) {
                
                }
                bool flag = false;
                if (flag == false) {

                }
                if (false == flag) {
                    
                }
       
           for (i = 0; i < n-1; i--)     
           {
                j++;
           }
    
        }
        """,
'''
void heapify(int arr[], int n, int i)
{
    int largest = i;   // Initialize largest as root
    int l = 2 * i + 1; // left = 2*i + 1
    int r = 2 * i + 2; // right = 2*i + 2

    // If left child is larger than root
    if (l < n && arr[l] > arr[largest])
        largest = l;

    // If right child is larger than largest so far
    if (r < n && arr[r] > arr[largest])
        largest = r;

    // If largest is not root
    if (largest != i)
    {
        swap(arr, i, largest);

        // Recursively heapify the affected sub-tree
        heapify(arr, n, largest);
    }
}'''
    ]

    for sample_code in sample_codes[1:]:
        # swap(sample_code)
        permute_codes, code = swap(sample_code)
        print("-" * 50)
        for i in permute_codes:
            print(i)
        print("-" * 50)
        print(code)
        break


if __name__ == "__main__":
    main()
