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
                    try:
                        statement, _ = list(reversed(_statements))[idx]
                    except:
                        print('-----')
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
#         """
# int _Partition(int *A, int left, int right)
# {
#     int pivot = _Pivot(A, left, right);
#     int i = left + 1, j = right - 2;
#     while (1)
#     {
#         while (A[i] < pivot)
#             i++;
#         while (A[j] > pivot)
#             j--;
#         if (i < j)
#         {
#             swap(A, i, j);
#             i++;
#             j--;
#         }
#         else
#             break;
#     }
#     swap(A, i, right - 1);
#     return i;
# }"""
'public void var1(ActionEvent var2) { URL var72 = new URL("getURL.cgi" + getCodeBase().toString()); StringBuffer var63 = new StringBuffer(""); String var30 = readURL(var29); var8 = true; boolean var46 = false; String var26 = ""; DataInputStream var64 = new DataInputStream(con.getInputStream()); int var14 = 0; for (int var14 = 0; 10 > var14; var14=var14+1) { if (var57[var14] == e.getSource()) { if (var58) printAnswer(); print.setVisible(true); selectTerm.setVisible(true); var13 = var14; textArea2.setCaretPosition(1 - textArea2.getText().length()); var10 = textArea2.getText().indexOf((var14 + 1) + "#"); if (-1 != var10) textArea2.setCaretPosition(var10); repaint(); } } if (var16 == e.getSource()) { if (\'#\' != textArea2.getText().charAt(0)) printAnswer(); DataOutputStream var74 = new DataOutputStream(con.getOutputStream()); int var54 = 0; var24 = data.indexOf("\\n", var24); var24+=1; DataInputStream var64 = new DataInputStream(new BufferedInputStream(yc.getInputStream())); var23 = data.substring(var24, var25); int var25 = data.indexOf("\\n---------", var24); if ("Spring" == term.getSelectedItem()) var26 = "SP"; else if ("Summer" == term.getSelectedItem()) var26 = "SU"; else var26 = "FL"; String var79; try { String var23 = textArea2.getText(); String var26; a.showDocument(var60, "_blank"); } catch (MalformedURLException var52) { } } if (var17 == e.getSource()) { int var12 = var13; if ("Spring" == term.getSelectedItem()) var26 = "SP"; else if ("Summer" == term.getSelectedItem()) var26 = "SU"; else var26 = "FL"; AppletContext var59 = getAppletContext(); String var28 = courseNum.getText().toUpperCase(); try { int var14 = 0; char var9; a.showDocument(var60, "_blank"); } catch (MalformedURLException var52) { } } if (var18 == e.getSource()) { printSign("Loading..."); int var14 = 0; var29 = fileName.replace(\' \', \'_\'); int var54 = 0; if (!publicSign.equals("Error loading.")) { textArea1.setText(var30); var29=var29+".2"; var30 = readURL(var29); absorb(var30); printAnswer(); for (String var29 = idField.getText(); 10 > var14; var14+=1) { if (var70[var14].var66 != 10000 && var70[var14].var66 != -1 && var70[var14].var66 != 9999) { var57[var14].setVisible(true); } else var57[var14].setVisible(false); } if (!var57[0].isVisible()) { print.setVisible(false); selectTerm.setVisible(false); } else { print.setVisible(true); selectTerm.setVisible(true); } printSign("Load complete."); } var13 = 0; repaint(); } if (var19 == e.getSource()) { String var34 = ""; var29 = fileName.replace(\' \', \'_\'); printSign("Saving..."); writeURL(var29, 1); printSign("Saving......"); var29=var29+".2"; writeURL(var29, 2); printSign("Save complete."); } if (var20 == e.getSource()) { showInstructions(); } if (var21 == e.getSource()) { var31 = false; int var54 = 0; String var26; String var78; URLConnection var73 = url.openConnection(); textArea2.setText("Retrieving Data..."); try { String var32 = ""; if ("Spring" == term.getSelectedItem()) var26 = "SP"; else if ("Summer" == term.getSelectedItem()) var26 = "SU"; else var26 = "FL"; StringTokenizer var69 = new StringTokenizer(var3, ","); var7 = lst.getSelectedItem().toString(); { var34 =  var28+"http://sis450.berkeley.edu:4200/OSOC/osoc?p_term=" ; try { String var27 =  var28+"http://sis450.berkeley.edu:4200/OSOC/osoc?p_term=" ; StringTokenizer var83 = new StringTokenizer(andST.nextToken()); con.setDoOutput(true); con.setDoInput(true); con.setUseCaches(false); con.setRequestProperty("Content-type", "application/x-www-form-urlencoded"); AppletContext var59 = getAppletContext(); int var80, var81, var82; out2.writeBytes(var75); out2.flush(); StringTokenizer var43 = new StringTokenizer(var42, "\\n"); int var11; while (null != (var27 = in.readLine())) { } in.close(); } catch (IOException var67) { } } URL var60 = new URL(var27); URLConnection var62 = yahoo.openConnection(); String var28 = courseNum.getText().toUpperCase(); String var35 = (parseData(var30, false)); while (null != (var32 = in.readLine())) { buf.append(var32); } var30 = buf.toString(); in.close(); } catch (IOException var53) { } public void var1 if (inText.equals("-1")) var35 = parseData(var30, true); if (inText.equals("\\n")) { textArea2.append( ")"+"\\nNO DATA FOUND \\n(" ); } else textArea1.append(var35); repaint(); } var15 = false; if (var22 == e.getSource()) { if (t.isAlive() && var51 != null) { t.stop(); epilogue(); return; } var13 = 0; textArea2.setCaretPosition(0); for (int var49 = 0; 30 > var14; var14+=1) for (int var24 = data.indexOf((var13 + 1) + "#"); 20 > var54; var54+=1) { var76[var14][var54] = new entry(); var76[var14][var54].var47 = new Time[4]; for (int var54 = 0; 4 > var49; var49=var49+1) { var76[var14][var54].var47[var49] = new Time(); var76[var14][var54].var47[var49].var77 = 0; } } var36 = new entry[30]; for (int var14 = 0; 30 > var14; var14=var14+1) { var36[var14] = new entry(); var36[var14].var47 = new Time[4]; for (URL var60 = new URL(var27); 4 > var54; var54=var54+1) { var36[var14].var47[var54] = new Time(); var36[var14].var47[var54].var77 = 0; } } var37 = -5; var38 = -1 * var55; var39 = 0; var40 = 0; var31 = false; var41 = ""; int var14 = 0; if (-1 == text1.toUpperCase().indexOf("OR:")) var56 = false; else var56 = true; var42 = removeOR(text1.toUpperCase()); String var27 =  URLEncoder.encode(var23)+getCodeBase().toString() ; var44 = -1; var45 = -1; String var27; while (st.hasMoreTokens()) { var3 = st.nextToken().toString(); if (line.equals("")) break; else var9 = line.charAt(0); if (\'0\' == var9) { var15 = true; repaint(); break; } if (var9 <= \'9\' && var9 >= \'1\') { var46 = false; var45+=1; if (30 == var45) { var58 = true; textArea2.setText("Error: Exceeded 30 time entries per class."); var15 = true; repaint(); return; } var68 = -1; String var3, var4; while (andST.hasMoreTokens()) { int var10; String var33; String var5, var6; var68+=1; if (4 == var68) { var58 = true; textArea2.setText("Error: Exceeded 4 time intervals per entry!"); var15 = true; repaint(); return; } int var14 = 0; var78 = timeST.nextToken().toString(); var79 = ""; var80 = 0; if (temp.equals("")) break; while (\'-\' != temp.charAt(var80)) { var79=var79+temp.charAt(var80); var80=var80+1; if (temp.length() <= var80) { var58 = true; textArea2.setText("Error: There should be no space before hyphens."); var15 = true; repaint(); return; } } try { var81 = Integer.parseInt(var79); } catch (NumberFormatException var85) { var58 = true; textArea2.setText("Error: There should be no a/p sign after FROM_TIME."); var15 = true; repaint(); return; } var80+=1; var79 = ""; if (temp.length() <= var80) { var15 = true; repaint(); var58 = true; textArea2.setText("Error: am/pm sign missing??"); return; } while (temp.charAt(var80) <= \'9\' && temp.charAt(var80) >= \'0\') { var79=var79+temp.charAt(var80); var80+=1; if (temp.length() <= var80) { var15 = true; repaint(); var58 = true; textArea2.setText("Error: am/pm sign missing??"); return; } } var82 = Integer.parseInt(var79); if (\'a\' == temp.charAt(var80) || \'A\' == temp.charAt(var80)) { } else { if (!timeEq(var82, 1200) && isLesse(var81, var82)) { if (4 == String.valueOf(var81).length() || 3 == String.valueOf(var81).length()) { var81=var81+1200; } else var81=var81+12; } if (!timeEq(var82, 1200)) { if (4 == String.valueOf(var82).length() || 3 == String.valueOf(var82).length()) { var82=var82+1200; } else var82=var82+12; } } if (2 == String.valueOf(var81).length() || 1 == String.valueOf(var81).length()) var81=var81*100; if (2 == String.valueOf(var82).length() || 1 == String.valueOf(var82).length()) var82=var82*100; var76[var45][var44].var47[var68].var77 = var81; var76[var45][var44].var47[var68].var86 = var82; if (timeST.hasMoreTokens()) var4 = timeST.nextToken().toString(); else { var58 = true; textArea2.setText("Error: days not specified?"); var15 = true; repaint(); return; } if (days.equals("")) return; if (-1 != days.indexOf("M") || -1 != days.indexOf("m")) var76[var45][var44].var47[var68].var87 = 1; if (-1 != days.indexOf("TU") || -1 != days.indexOf("Tu") || -1 != days.indexOf("tu")) var76[var45][var44].var47[var68].var88 = 1; if (-1 != days.indexOf("W") || -1 != days.indexOf("w")) var76[var45][var44].var47[var68].var89 = 1; if (-1 != days.indexOf("TH") || -1 != days.indexOf("Th") || -1 != days.indexOf("th")) var76[var45][var44].var47[var68].var90 = 1; if (-1 != days.indexOf("F") || -1 != days.indexOf("f")) var76[var45][var44].var47[var68].var91 = 1; } } else { if (var46) var44=var44-1; var44+=1; if (20 == var44) { var58 = true; textArea2.setText("Error: No more than 20 class entries!"); var15 = true; repaint(); return; } var45 = -1; var3 = line.trim(); for (String var29 = idField.getText(); 30 > var14; var14=var14+1) var76[var14][var44].var71 = var3; var46 = true; } } for (URL var61 = new URL(this.getCodeBase(), "classData.txt"); 30 > var14; var14+=1) { for (String var75 = URLEncoder.encode(var34) + "url="; 4 > var54; var54=var54+1) { var36[var14].var47[var54].var77 = 0; } } for (String var30 = ""; 10 > var14; var14=var14+1) { var65[var14] = 10000; var70[var14].var66 = 10000; for (String var7 = ""; 30 > var54; var54+=1) var70[var14].var84[var54].var71 = ""; } var47 = 0; var48 = 0; String var42 = textArea1.getText(); calculateTotalPercent(0, "\\n"); var50 = var48; button1.setLabel("...HALT GENERATION..."); printWarn(); if (t.isAlive() && var51 != null) t.stop(); var51 = new Thread(this, "Generator"); t.start(); } }'
    ]

    for sample_code in sample_codes:
        # swap(sample_code)
        permute_codes, code = whileForExchange(sample_code)
        print("-" * 50)
        for i in permute_codes:
            print(i)
        print("-" * 50)
        # print(code)
        break


if __name__ == "__main__":
    main()
