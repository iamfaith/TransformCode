from os import path
from tree_sitter import Language, Parser
from pathlib import Path
import glob, os
import numpy as np
import logging
import urllib.request
from urllib3.exceptions import InsecureRequestWarning
from tqdm import tqdm
import zipfile
import shutil
import platform

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
           ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

class ASTParser():
    import logging
    LOGGER = logging.getLogger('ASTParser')
    def __init__(self, language='java'):
        # ------------ To initialize for the treesitter parser ------------
        home = str(Path.home())
        cd = os.getcwd()
        plat = platform.system()     
        p = path.join(home, ".tree-sitter")
        # os.chdir(path.join(p, "tree-sitter-parsers-" + plat))
        os.chdir(path.join(p, "bin"))
        self.languages = {}
        #  download
        if not path.exists(p):
            os.makedirs(p, exist_ok=True)
            zip_url = "https://github.com/yijunyu/tree-sitter-parsers/archive/refs/heads/" + plat + ".zip"
            parsers_target = os.path.join(p, plat + ".zip")
            try:
                # download from precompiled binaries
                download_url(zip_url, parsers_target)
                with zipfile.ZipFile(parsers_target, 'r') as zip_ref:
                    zip_ref.extractall(p)
            except: 
                plat = "main"
                # build from scratch
                langs = []
                os.chdir(path.join(p, "tree-sitter-parsers-" + plat))
                for file in glob.glob("tree-sitter-*"):        
                    lang = file.split("-")[2]
                    if not "." in file.split("-")[3]: # c-sharp => c_sharp.so
                        lang = lang + "_" + file.split("-")[3]
                    langs = [file]
                    Language.build_library(
                        # Store the library in the `build` directory
                        lang + '.so',
                        # Include one or more languages
                        langs
                    )

        for file in glob.glob("*.so"):
            try:
                lang = os.path.splitext(file)[0]
                # self.Languages[lang] = Language(path.join(p, "tree-sitter-parsers-" + plat, file), lang)
                self.languages[lang] = Language(path.join(p, "bin", file), lang)
            except Exception as e:
                print("An exception occurred to {}".format(lang), e)
        os.chdir(cd)
        self.parser = Parser()

        self.language = language
        if self.language == None:
            logging.info(
                "Cannot find language configuration, using java parser as the default to parse the code into AST")
            self.language = "java"

        lang = self.languages.get(self.language)
        self.parser.set_language(lang)
        # -----------------------------------------------------------------

       
    def parse_with_language(self, code_snippet, language):
        lang = self.languages.get(language)
        self.parser.set_language(lang)
        return self.parser.parse(code_snippet)

    def parse(self, code_snippet):
        return self.parser.parse(code_snippet)
    
    def set_language(self, language):
        lang = self.languages.get(language)
        self.parser.set_language(lang)
        return lang

