try:
    from base_operator import BaseOperator
except:
    from .base_operator import BaseOperator


class VariableRenaming(BaseOperator):
    def __init__(self, language: str):
        super(VariableRenaming, self).__init__(language)
        self.var_node_types = {'identifier'}
        self.var_filter_types = {'class_declaration', 'method_declaration', 'method_invocation', 'function_declarator'}
        # 'variable_declarator'
        if "c" == language or "cpp" == language:
            self.var_filter_types.add("call_expression")

    # Get only variable node from type "identifier"
    def get_identifier_nodes(self, tree, text):
        var_nodes, var_renames = [], {}
        queue = [tree.root_node]
        while queue:
            current_node = queue.pop(0)
            for child_node in current_node.children:
                child_type = str(child_node.type)
                if child_type in self.var_node_types:  # only identifier node
                    if str(current_node.type) in self.var_filter_types:
                        # filter out class/method name or function call identifier
                        continue
                    var_name = text[child_node.start_byte: child_node.end_byte]
                    if var_name not in var_renames:
                        var_renames[var_name] = "var{}".format(len(var_renames) + 1)
                    var_nodes.append([child_node, var_name, var_renames[var_name]])
                queue.append(child_node)
        return var_nodes

    def transform(self, id_nodes, code_text):
        id_nodes = sorted(id_nodes, reverse=True, key=lambda x: x[0].start_byte)
        for var_node, var_name, var_rename in id_nodes:
            code_text = code_text[:var_node.start_byte] + var_rename + code_text[var_node.end_byte:]
        return code_text

    def variableRenaming(self, code_snippet):
        tree = self.parse(code_snippet)
        identifier_nodes = self.get_identifier_nodes(tree, code_snippet)
        return identifier_nodes, self.transform(identifier_nodes, code_snippet)


def main():
    var_renaming_operator = VariableRenaming(language="java")

    sample_codes = [
        """
        public class Main {
          public static void main(String[] args) {
            int a = 15;
            int b = 20;
            int c = a + f(b);
          }
        }
        """,
        """
        void bubbleSort(int arr[], int n)
        {
           int i, j;
           for (i = 0; i < n-1; i++)     
         
               // Last i elements are already in place  
               for (j = 0; j < n-i-1; j++)
                   if (arr[j] > arr[j+1])
                      swap(&arr[j], &arr[j+1]);
        }
        """
    ]

    for sample_code in sample_codes:
        var_renaming_code = var_renaming_operator(sample_code)
        print(var_renaming_code)
        print("-" * 50)


if __name__ == "__main__":
    main()
