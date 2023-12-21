
import tree_sitter
from tree_sitter import Node
from typing import List
try:
    from ast_parser import ASTParser
    from base_operator import BaseOperator
except:
    from .base_operator import BaseOperator
    from .ast_parser import ASTParser
from collections import OrderedDict, Callable
from copy import deepcopy

class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
                not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               OrderedDict.__repr__(self))


class ExtractExecution(BaseOperator):

    def __init__(self, language: str):
        super(ExtractExecution, self).__init__(language)
        self.filterType = {
            'java': [
                "program",
                "ERROR",  # java
                "(",
                ")",
                "{",
                "}",
                ":",
                ";",
                ".",
                ",",
                "object_creation_expression",
                "block",
                "local_variable_declaration",
                "method_invocation",
                "new",
                "while_statement",
                "expression_statement",
                "if_statement",
                "try_statement",
                "continue_statement",
                "break_statement",
                "throw_statement",
                # "catch_clause",
                "catch_formal_parameter",
                "try",
                "catch",
                "catch_type",
                #############################
                "assignment_expression",
                "parenthesized_expression",
                "binary_expression",
                "update_expression",
                #############################
                "variable_declarator",
                "modifiers",  # private public
                # "argument_list",  # need to check
                "formal_parameters",
                #################### update: add
                "return",
                "int",
                "integral_type",
                #################### update: add
                # "identifier",
                # "string_literal", # needed
                # "field_access"
            ],
            'python': [
                "module",  # python
                '"',  # python  comment """
                # 'string',  # python ("'''\n.git/\n'''", 'string', <Node type=string,
                '(',
                ')',
                ',',
                # ('labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids', 'expression_statement'
                'expression_statement',
                'ERROR'
            ],
            'c': [
                "program",
                "ERROR",  # java
                "(",
                ")",
                "{",
                "}",
                ":",
                ";",
                ".",
                ",",
                "object_creation_expression",
                "block",
                "local_variable_declaration",
                "method_invocation",
                "new",
                "while_statement",
                "expression_statement",
                "if_statement",
                "try_statement",
                "continue_statement",
                "break_statement",
                "throw_statement",
                # "catch_clause",
                "catch_formal_parameter",
                "try",
                "catch",
                "catch_type",
                #############################
                "assignment_expression",
                "parenthesized_expression",
                "binary_expression",
                "update_expression",
                #############################
                "variable_declarator",
                "modifiers",  # private public
                # "argument_list",  # need to check
                "formal_parameters",
                #################### update: add
                # "return",
                "int",
                "integral_type",
                #################### update: add
            ],
        }
        self.captures = ["method_declaration"]
        if "c" == language or "cpp" == language:
            self.captures = ["function_definition"]
        self.currentFilter = self.filterType[language]
        self.debug = False
        # self.debug = True

    def loopNode(self, code, current_node, return_node=False):
        def _cb(child, token):
            # not loop into
            # update: add "return_statement", "formal_parameter"
            if self.getNodeType(child) in ["field_access", "array_access", 
                                           "return_statement",
                                            # "catch_clause",
                                            "subscript_expression", ### c
                                            "argument_list",
                                           "scoped_type_identifier", "formal_parameter"]:
                # print("---", child, token)
                return False
            return True

        nodes = self.get_nodes(code, current_node, ignore_types=None, capture_types=None, cb=_cb, focus=["subscript_expression", "argument_list"])
        ret = []
        custom_filter = deepcopy(self.currentFilter)
        custom_filter.append("block")
        custom_filter.append("while_statement")
        custom_filter.append("for_statement")
        custom_filter.append("if_statement")
        custom_filter.append("catch_clause")

        # for c/cpp
        custom_filter.append("call_expression")
        custom_filter.append("compound_statement")

        # custom_filter.remove("parenthesized_expression")
    
        for idx, (node, token) in enumerate(nodes):
            nodeType = self.getNodeType(node)
            if nodeType in custom_filter:
                continue
            if self.debug:
                print(idx, token, nodeType)
            idx += 1
            if return_node:
                ret.append((token, node))
            else:
                ret.append(token)
        return ret

    def _extract_method_node(self, code_snippet, method):
        statements = self.get_nodes(
            code_snippet, method, debug=self.debug, ignore_types=["comment"],
            capture_types=["for_statement", "while_statement", "assignment_expression",
                            "return_statement",
                           "local_variable_declaration", "expression_statement", "ERROR",
                           "if_statement"])
        tokens = []
        idx = 0
        for _, (node, token) in enumerate(statements):
            if node.start_byte == 0 and self.getNodeType(node) == 'local_variable_declaration':
                continue
            nodeType = self.getNodeType(node)
            # if nodeType in self.currentFilter:
            #     continue
            if self.debug:
                print(idx, token, nodeType)
            idx += 1
            tokens.extend(self.loopNode(code_snippet, node, return_node=True))
        return tokens

    def _check_method(self, _tokens, methods_dict, methods_dict_nodes, tokens, code_snippet):
        # check tokens
        for _token_tmp, _node_tmp in _tokens:
            if self.getNodeType(_node_tmp) == "identifier" and _token_tmp in methods_dict:
                method_node = methods_dict_nodes[_token_tmp]
                _tokens = self._extract_method_node(code_snippet, method_node)
                del methods_dict[_token_tmp]
                self._check_method(_tokens, methods_dict, methods_dict_nodes, tokens, code_snippet)
            else:
                tokens.append(_token_tmp)

    def extractClassExecutionNew(self, code_snippet):
        tree = self.parse(code_snippet)
        methods = self.get_nodes(code_snippet, tree.root_node, debug=self.debug, ignore_types=[
                                 "comment"], capture_types=self.captures)
        if len(methods) <= 1:
            return self.extractClassExecution(code_snippet)
        ret = []
        all_methods = []
        methods_dict = DefaultOrderedDict(set)
        methods_dict_nodes = OrderedDict()
        methods_names = set()
        for method_idx, (method, token) in enumerate(methods):
            method_name = self.get_nodes(code_snippet, method, debug=self.debug, ignore_types=[
                                         "comment"], capture_types=["identifier"])
            if len(method_name) > 0:
                _method_node, _method = method_name[0]
                all_methods.append(_method)
                methods_names.add(_method)
                for node, token in method_name[1:]:
                    methods_dict[_method].add(token)
                    methods_dict_nodes[_method] = method

        for key in methods_dict:
            methods_dict[key] = methods_dict[key] & methods_names

        max_method_cnt, main_method = 0, None
        for key in methods_dict:
            if len(methods_dict[key]) > max_method_cnt:
                max_method_cnt = len(methods_dict[key])
                main_method = key
        if main_method is not None:
            main_method_node = methods_dict_nodes[main_method]
            statements = self.get_nodes(
                code_snippet, main_method_node, debug=self.debug, ignore_types=["comment"],
                capture_types=["for_statement", "while_statement", "assignment_expression",
                            "local_variable_declaration", "expression_statement", "ERROR",
                            "if_statement"])
            tokens = []
            idx = 0
            for i, (node, token) in enumerate(statements):
                if node.start_byte == 0 and self.getNodeType(node) == 'local_variable_declaration':
                    continue
                nodeType = self.getNodeType(node)
                # if nodeType in self.currentFilter:
                #     continue
                if self.debug:
                    print(idx, token, nodeType)
                idx += 1
                selected_nodes = self.loopNode(code_snippet, node, return_node=True)
                for _token, _node in selected_nodes:
                    if self.getNodeType(_node) == "identifier" and _token in methods_dict:
                        # add method into currment method
                        method_node = methods_dict_nodes[_token]
                        _tokens = self._extract_method_node(code_snippet, method_node)
                        tokens.append(_token)
                        del methods_dict[_token]

                        self._check_method(_tokens, methods_dict, methods_dict_nodes, tokens, code_snippet)
                        
                        # tokens.extend(_tokens)
                    else:
                        tokens.append(_token)
            if len(tokens) > 0:
                ret.extend(tokens)
        else:
            # print('----')
            for key in methods_dict_nodes:
                # print(key)
                search_node = methods_dict_nodes[key]
                #################### update: add "if_statement"
                statements = self.get_nodes(
                    code_snippet, search_node, debug=self.debug, ignore_types=["comment"],
                    capture_types=["for_statement", "while_statement", "assignment_expression",
                                "local_variable_declaration", "expression_statement", "ERROR",
                                "if_statement"])

                tokens = []
                for idx, (node, token) in enumerate(statements):
                    if node.start_byte == 0 and self.getNodeType(node) == 'local_variable_declaration':
                        continue
                    nodeType = self.getNodeType(node)
                    # if nodeType in self.currentFilter:
                    #     continue
                    if self.debug:
                        print(idx, token, nodeType)
                    if self.debug:
                        print('-------------')
                    tokens.extend(self.loopNode(code_snippet, node))
                    if self.debug:
                        print('-------------')
                if len(tokens) > 0:
                    if len(tokens) > len(ret):
                        ret = tokens
                    # ret.extend(tokens)

        return ret

    def extractClassExecution(self, code_snippet):
        tree = self.parse(code_snippet)
        methods = self.get_nodes(code_snippet, tree.root_node, debug=self.debug, ignore_types=[
                                 "comment"], capture_types=self.captures)
        ret = []
        for method_idx, (method, token) in enumerate(methods):
            statements = self.get_nodes(
                code_snippet, method, debug=self.debug, ignore_types=["comment"],
                capture_types=["for_statement", "while_statement", "assignment_expression",
                               "local_variable_declaration", "expression_statement", "ERROR",
                               "if_statement"])
            tokens = []
            idx = 0
            for _, (node, token) in enumerate(statements):
                if node.start_byte == 0 and self.getNodeType(node) == 'local_variable_declaration':
                    continue
                nodeType = self.getNodeType(node)
                # if nodeType in self.currentFilter:
                #     continue
                if self.debug:
                    print(idx, token, nodeType)
                idx += 1
                tokens.extend(self.loopNode(code_snippet, node))
            # print(tokens)
            if len(tokens) > 0:
                # tokens.insert(0, f"method{method_idx + 1}")
                ret.extend(tokens)
        return ret

    def extractExecution(self, code_snippet):
        tree = self.parse(code_snippet)
        # statements = self.get_nodes(code_snippet, tree.root_node, debug=True, ignore_types=self.currentFilter)

        #################### update: add "if_statement"
        statements = self.get_nodes(
            code_snippet, tree.root_node, debug=self.debug, ignore_types=["comment"],
            capture_types=["for_statement", "while_statement", "assignment_expression",
                           "local_variable_declaration", "expression_statement", "ERROR",
                           "if_statement"])

        tokens = []
        idx = 0
        for _, (node, token) in enumerate(statements):
            if node.start_byte == 0 and self.getNodeType(node) == 'local_variable_declaration':
                continue
            nodeType = self.getNodeType(node)
            # if nodeType in self.currentFilter:
            #     continue
            if self.debug:
                print(idx, token, nodeType)
            idx += 1

            if self.debug:
                print('-------------')
            tokens.extend(self.loopNode(code_snippet, node))
            if self.debug:
                print('-------------')
            # break
        return tokens


if __name__ == "__main__":

    code = 'private void var2(String var3) throws var1 { String var4 = this.var7 + var3 + ".json"; Scanner var5 = new Scanner(new File(var4)); PrintWriter var6 = new PrintWriter(new File(var4 + ".new")); while (s.hasNextLine()) { fw.println(s.nextLine().replaceAll("NODEKEY", this.var8)); } s.close(); fw.close(); (new File(var4 + ".new")).renameTo(new File(var4)); }'

    source = 'public static String var1(String var2) { MessageDigest var3 = null; try { var3 = MessageDigest.getInstance("MD5"); md5.reset(); md5.update(value.getBytes()); } catch (NoSuchAlgorithmException var7) { log.error("Could not find the requested hash method: " + e.getMessage()); } byte[] var4 = md5.digest(); StringBuffer var5 = new StringBuffer(); for (int var6 = 0; var6 < var4.var8; var6++) { hexString.append(Integer.toHexString(0xFF & var4[var6])); } return hexString.toString(); }'

    code = source
    # code = target
    code = 'public static void var1(String var2) { try {\nlogger.info(var2 + "Address: ");\n} catch (Exception ex) {\nlogger.info(var8);\n}\nif (false) { int temp = 100; } try { URLConnection var5 = url.openConnection(); public static void var1 int var6 = 0;while(){ String var8 = conn.getHeaderField(var6); String var7 = conn.getHeaderFieldKey(var6); if (null == var8 && null == var7) { break; } if (null == var7) { try {\nex.printStackTrace();\n} catch (Exception ex) {\nex.printStackTrace();\n}\nif (false) { int temp = 100; } continue; } try {\nvar6++;\n} catch (Exception ex) {\nex.printStackTrace();\n}\nif (2 > 1) { try {\nlogger.info( var8+var7 );\n} catch (Exception ex) {\nex.printStackTrace();\n} } System.out.println("Dummy");} } catch (Exception var3) { logger.error("Excepe.getMessage() + "Exception Message: "'

    # code = 'public static void var1(String var2) { logger.info("Address: " + var2); try { URL var4 = new URL(var2); URLConnection var5 = url.openConnection(); for (int var6 = 0; ; var6++) { String var7 = conn.getHeaderFieldKey(var6); String var8 = conn.getHeaderField(var6); if (var7 == null && var8 == null) { break; } if (var7 == null) { logger.info(var8); continue; } logger.info(var7 + " " + var8); } } catch (Exception var3) { logger.error("Exception Message: " + e.getMessage()); } }'
    code = 'private void var4(int var8) { String var1 = var2.var3(this, "Skriv navn for å komme på highscorelisten!", "Lagre score!", JOvar8Nvar9 URvar5rvar6 urvar7ew URL(score=" + scvar10 URLConnection urvar7 url.openConnection(); urlConnection.connect(); urlConnection.getInputStream(); BrowserControl.openUrl("&navar12 + "http://129.177.17.51:8080/GuestBook/TheOnlyServlet?name="http://129.177.17.51:8080/GuestBook/TheOnlyServlet"); } catch (MalformedURLException e)var11avar9e.printStackTrace(); } catch (IOvar10)var11avar9e.printStackTrace(); } }'

    code = '''public var1(URL var3, String var4, Map<String, String> var5, String var6) throws var2 { ex.printStackTrace();\nif (1 > 2)  { try {\nSystem.out.println("Dummy2");\n} catch (Exception ex) {\nlogger.debug( var5+"Creating http url handler for: " );\n} } else  { System.out.println("Dummy1"); } if (var3 == null) throw new IllegalArgumentException("Null poinnull == var3; if (!"http".equals(url.getProtocol()) && !"https".equals(url.ge!"https".equals(url.getProtocol()) && !"http".equals(url.getProtocol()) or \\"https\\""+"Illegal url protocol: \\"" ); if (var4 == null) throw new IllegalArgumentException("Null pointer in requestMethodnull == var4T".equals(var4) && !"POST".equals(var4)) throw new IllegalArgumentException!"POST".equals(var4) && !"GET".equals(var4)request method: " ); if (var5 == null) throw new IllegalArgumentException("Null pointer in parameters"); HttpURnull == var5var7 = (HttpURLConnection) url.openConnection(var10); connection.setRequestMethod(var4); connection.setUseCaches(false); if (EMPTY_MAP.equals(var5)) { connection.setDoOutput(false); } else { connection.setDoOutput(true); OutputStream var9 = connection.getOutputStream(); writeParameters(var9, var5, var6); out.close(); } var8 = connection.getInputStream(); }'''

    code = '''public void getMax(int a, int b) { if (a>b) return a; else return b;}'''
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
    # code = '''public void getMax(int a, int b) { if (b<a) return a; else return b;}'''
    extract = ExtractExecution('java')
    t = extract.extractClassExecutionNew(code)
    print(t)
    # print(''.join(t))
