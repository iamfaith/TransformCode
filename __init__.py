

import ast
try:
    from comment_deletion import CommentDeletion as CommentDeletion
    from permute_declaration import PermuteDeclaration
    from permute_statement import PermuteStatement   # need to test
    from swap_condition import SwapCondition
    from var_renaming import VariableRenaming
    from while_for_exchange import WhileForExchange
    from add_dummy_statement import AddDummyStatement
    from add_try_catch import AddTryCatch
    from extract_execution import ExtractExecution
    from arithmetic_transform import ArithmeticTransform
    from method_name_extract import MethodNameExtract
except:
    from .comment_deletion import CommentDeletion as CommentDeletion
    from .permute_declaration import PermuteDeclaration
    from .permute_statement import PermuteStatement   # need to test
    from .swap_condition import SwapCondition
    from .var_renaming import VariableRenaming
    from .while_for_exchange import WhileForExchange
    from .add_dummy_statement import AddDummyStatement
    from .add_try_catch import AddTryCatch
    from .extract_execution import ExtractExecution
    from .arithmetic_transform import ArithmeticTransform
    from .method_name_extract import MethodNameExtract
import os

global_lang = os.environ.get("global_lang", None)
print("global_lang", global_lang)


def extractMethodName(code, lang='java'):
    if global_lang is not None:
        lang = global_lang
    extract = MethodNameExtract(language=lang)
    method_code_pairs = extract(code)
    ret = []
    for method_name, method_code in method_code_pairs:
        _method_code = normalizeCode(method_code, lang=lang)
        
        # code_path = extractCodePath(_method_code, lang=lang, method="extractClassExecutionNew")
        code_path = extractCodePath(_method_code, lang=lang)
        # if code_path == '':
            # code_path = extractCodePath(_method_code, lang=lang)
        code_path = code_path.replace('\n', '')
        code_path = code_path.replace('\t', '')
        # code_path = extractCodePath(_method_code, lang=lang)

        _trans_code = transformCode(_method_code, lang=lang)
        trans_code_path = extractCodePath(_trans_code, lang=lang)

        # _trans_code = transformCode(_method_code, lang=lang, suffix='Class')
        # trans_code_path = extractCodePath(_trans_code, lang=lang, method="extractClassExecutionNew")

        # if trans_code_path == '':
            # trans_code_path = extractCodePath(_trans_code, lang=lang)
        trans_code_path = trans_code_path.replace('\n', '')
        trans_code_path = trans_code_path.replace('\t', '')
        # trans_code_path = extractCodePath(_trans_code, lang=lang)

        if method_code == '' or code_path == '' or trans_code_path == '':
            continue

        # print(method_name, code_path)
        # print(trans_code_path)
        ret.append((method_name, code_path, trans_code_path, method_code))
    return ret
    

def extractCodePath(code, lang='java', method=None):

    if global_lang is not None:
        lang = global_lang
    # print("extract", lang)
    extract = ExtractExecution(language=lang)
    if method is not None and hasattr(extract, method) and lang == 'java':
        token = getattr(extract, method)(code)
    else:
        token = extract(code)
    # print(token)
    return ' '.join(token)


def normalizeCode(code, lang='java', delComment=True):
    if global_lang is not None:
        lang = global_lang
    code = code.replace('  ', '')
    if delComment:
        commentDeletion = CommentDeletion(language=lang)
        _, code = commentDeletion(code)
        rename = VariableRenaming(language=lang)
        _, code = rename(code)
    return code


def transformCode(
        code,
        transformChain=[PermuteDeclaration, SwapCondition, ArithmeticTransform,
                        WhileForExchange, AddDummyStatement, AddTryCatch, PermuteStatement],
        suffix='',
        lang="java"):
    if global_lang is not None:
        lang = global_lang
    code = normalizeCode(code, lang=lang)
    if len(transformChain) == 7 and ("c" == lang or "cpp" == lang):
        transformChain.pop(-2)
        ####### pop AddTryCatch PermuteStatement
        transformChain.pop(-1)
    # transformChain = [PermuteStatement]
    for Transform in transformChain:
        trans = Transform(language=lang)
        _, code = trans(code, suffix=suffix)
        # print('----', trans.__class__.__name__)
        # print(code)
        # print('----')
    return code

######################## not used
# def get_execution_path(code):
#     # Parse the Python code into an AST
#     tree = ast.parse(code)

#     # Find all functions and methods in the code
#     functions = [node for node in ast.walk(tree) if isinstance(
#         node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]

#     # Extract the execution paths from each function or method
#     paths = []
#     for func in functions:
#         path = []
#         current_node = func
#         while current_node is not None:
#             if isinstance(
#                     current_node, ast.FunctionDef) or isinstance(
#                     current_node, ast.AsyncFunctionDef):
#                 # Add the function or method name to the path
#                 path.append(current_node.name)

#             # Find the next control flow statement
#             next_node = None
#             for child in ast.iter_child_nodes(current_node):
#                 if isinstance(child, ast.If):
#                     next_node = child
#                     break
#                 elif isinstance(child, ast.While):
#                     next_node = child
#                     break
#                 elif isinstance(child, ast.For):
#                     next_node = child
#                     break
#                 elif isinstance(child, ast.Try):
#                     next_node = child
#                     break
#                 elif isinstance(child, ast.With):
#                     next_node = child
#                     break
#                 elif isinstance(child, ast.Raise):
#                     next_node = child
#                     break
#                 elif isinstance(child, ast.Return):
#                     next_node = None  # end of function
#                     break

#             if next_node is None:
#                 # Add the end of function marker to the path
#                 path.append("[END]")
#             else:
#                 # Add the control flow statement to the path and continue
#                 if isinstance(next_node, ast.If):
#                     path.append("if")
#                 elif isinstance(next_node, ast.While):
#                     path.append("while")
#                 elif isinstance(next_node, ast.For):
#                     path.append("for")
#                 elif isinstance(next_node, ast.Try):
#                     path.append("try")
#                 elif isinstance(next_node, ast.With):
#                     path.append("with")
#                 elif isinstance(next_node, ast.Raise):
#                     path.append("raise")

#             current_node = next_node

#         # Combine the path elements into a single string with a separator
#         path_separator = " -> "
#         paths.append(path_separator.join(path))

#     # Combine the execution paths of all functions into a single string
#     execution_path = " -> ".join(paths)

#     return execution_path


if __name__ == "__main__":
    code = '''
public class BubbleSortExample {  
    static void bubbleSort(int[] arr) {  
        int n = arr.length;  
        int temp = 0;  
        for(int i=0; i < n; i++) {  
            for(int j=1; j < (n-i); j++) {
                if(arr[j-1] > arr[j]){
                    // swap elements  
                    temp = arr[j-1];  
                    arr[j-1] = arr[j];  
                    arr[j] = temp;  
                }            
            }  
        }
    }
}
'''
    # code = normalizeCode(code)
    # print(code)
    # print('----')

# ['n', '=', 'arr.length', 'temp', '=', '0', 'for', 'i', '<', 'n', 'i', '++', 'i', '=', '0', 'for', 'j', '<', 'j', '++', 'j', '=', '1', 'if', 'n', '-', 'i', 'arr[j-1]', '>', 'arr[j]', 'temp', '=', 'arr[j-1]', 'arr[j-1]', '=', 'arr[j]', 'arr[j]', '=', 'temp']

# ['var2', '=', 'var1.var5', 'var3', '=', '0', 'for', 'var4', '<', 'var2', 'var4', '++', 'var4', '=', '0', 'for', 'var6', '<', 'var6', '++', 'var6', '=', '1', 'if', 'var2', '-', 'var4', 'var1[var6-1]', '>', 'var1[var6]', 'var3', '=', 'var1[var6-1]', 'var1[var6-1]', '=', 'var1[var6]', 'var1[var6]', '=', 'var3']
    code = '''
package com.badlogic.gdx.backends.gwt.preloader;

public interface LoaderCallback<T> {
        public void success (T result);

        public void error ();
}
    '''
    code = extractCodePath(code, method="extractClassExecutionNew")
    print(code, len(code))

    filename = "/home/faith/java-small/validation/libgdx/LoaderCallback.java"
    filename = '/home/faith/java-small/validation/libgdx/RenderableSorter.java'
    with open(filename, "r") as f:
        code = f.read()
        print(extractMethodName(code))

#     code = '''
# def min(a,b): 
#     if a<b: 
#         return a 
#     else:
#         return b
# '''
#     code = '''public void getMax(int a, int b) { if (a>b) return a; else return b;}'''
#     # path = get_execution_path(code)
#     path = extractCodePath(code)
#     print(path)
