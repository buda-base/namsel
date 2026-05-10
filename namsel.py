#! /usr/bin/env python
# encoding: utf-8
import logging
import sys
from PIL import Image
from config_manager import Config, default_config
from config_util import load_config, save_config
from line_breaker import LineCluster, LineCut
import numpy as np
from page_elements2 import PageElements as PE2
from recognize import cls, rbfcls
from recognize import recognize_chars_probout, recognize_chars_hmm, \
    viterbi_post_process, hmm_recognize_bigram
from segment import Segmenter, combine_many_boxes
import argparse
from utils_extra.scantailor_multicore import run_scantailor
from fast_utils import fadd_padding
import codecs
import os
from yik import word_parts_set
from root_based_finder import is_non_std
from termset import syllables
import tempfile
from subprocess import check_call

class FailedPageException(Exception):
    pass

class PageRecognizer(object):
    def __init__(self, imagefile, conf, page_info={}):
        confpath = conf.path
        self.conf = conf.conf
        self.imagefile = imagefile
        self.page_array = np.asarray(Image.open(imagefile).convert('L'))/255
        if self.page_array.all():
            self.conf['line_break_method'] = 'line_cut'

        # Determine whether a page is of type book or pecha        
        # Define line break method and page type if needed
        self.line_break_method = self.conf['line_break_method']
        self.page_type = self.conf['page_type']
        self.retries = 0
        self.page_info = page_info
        
        self.imgheight = self.page_array.shape[0]
        self.imgwidth = self.page_array.shape[1]
        
        # Determine line break method and page_type if not specified
        if not self.line_break_method and not self.page_type:
            if self.page_array.shape[1] > 2*self.page_array.shape[0]:
                print 'Setting page type as pecha'
                self.line_break_method = 'line_cluster'
                self.page_type = 'pecha'
            else: 
                print 'setting page type as book'
                self.line_break_method = 'line_cut'
                self.page_type = 'book' 

        self.conf['page_type'] = self.page_type
    
        self.conf['line_break_method'] = self.line_break_method
        if self.line_break_method == 'line_cluster' and self.page_type != 'pecha':
            print 'Must use page_type=pecha with line_cluster. Changing page_type'
            self.page_type = 'pecha'
        self.detect_o = self.conf.get('detect_o', False)

    ################################
    # The main recognition pipeline
    ################################
    def get_page_elements(self):
        '''PageElements (PE2) does a first-pass segmentation of blob (characters/punc)
        ona page, gathers information about width of page objects,
        isolates body text of pecha-style pages, and determines the
        number of lines on a page for use in line breaking'''
        self.shapes = PE2(self.page_array, cls, page_type=self.page_type, 
                     low_ink=self.conf['low_ink'], 
                     flpath=self.page_info.get('flname',''),
                     detect_o=self.detect_o, 
                     clear_hr =  self.conf.get('clear_hr', False))
        self.shapes.conf = self.conf
        if self.page_type == 'pecha' or self.line_break_method == 'line_cluster':
            if not hasattr(self.shapes, 'num_lines'):
                print 'Error. This page can not be processed. Please inspect the image for problems'
                raise FailedPageException('The page ({}) you are attempting to process failed'.format(self.imagefile))
            self.k_groups = self.shapes.num_lines
            self.shapes.viterbi_post = self.conf['viterbi_postprocess']

    def extract_lines(self):
        '''Identify lines on a page of text'''
        if self.line_break_method == 'line_cut':
            self.line_info = LineCut(self.shapes)
            if not self.line_info: # immediately skip to re-run with LineCluster
                sys.exit()
        elif self.line_break_method == 'line_cluster':
            self.line_info = LineCluster(self.shapes, k=self.k_groups)

        self.line_info.rbfcls = rbfcls
            
    def generate_segmentation(self):  
        self.segmentation = Segmenter(self.line_info)
    
    def recognize_page(self, text=False):
        try:
            self.get_page_elements()
            self.extract_lines()
        except:
            import traceback;traceback.print_exc()
            self.results = []
            return self.results
        
        self.generate_segmentation()
        
        conf = self.conf
        results = []
        try:
            if not conf['viterbi_postprocessing']:
                if conf['recognizer'] == 'probout':
                    results = recognize_chars_probout(self.segmentation)
                elif conf['recognizer'] == 'hmm':
                    results = recognize_chars_hmm(self.segmentation)
    
                if conf['postprocess']:
#                     print 'running viterbi post processing as next iter'
                    results = self.viterbi_post_process(self.page_array, results)
            else: # Should only be call from *within* a non viterbi run...
               # print 'Debug: Running within viterbi post proc'
                prob, results = hmm_recognize_bigram(self.segmentation)
    
                return prob, results
                
            output  = []
            for n, line in enumerate(results):
                for m,k in enumerate(line):
                    if isinstance(k[-1], int):
                        print n,m,k
                        self.page_array[k[1]:k[1]+k[3], k[0]:k[0]+k[2]] = 0
                        Image.fromarray(self.page_array*255).show()
                        
                    output.append(k[-1])
    
                output.append(u'\n')
    
            out =  ''.join(output)
            print out
        
            if text:
                results = out
            
            self.results = results
            return results
        except:
            import traceback;traceback.print_exc()
            if not results and not conf['viterbi_postprocessing']:
                print 'WARNING', '*'*40
                print self.page_info['flname'], 'failed to return a result.'
                print 'WARNING', '*'*40
                print
                if self.line_break_method == 'line_cut' and self.retries < 1:
                    print 'retrying with line_cluster instead of line_cut'
                    try:
                        pr = PageRecognizer(self.imagefile, Config(path=self.confpath, line_break_method='line_cluster', page_type='pecha'), page_info=self.page_info, retries = 1, text=text)
                        return pr.recognize_page()
                    except:
                        logging.info('Exited after failure of second run.')
                        return []
            if not conf['viterbi_postprocessing']: 
                if not results:
                    logging.info('***** No OCR output for %s *****' % self.page_info['flname'])
                if text:
                    results = out
                self.results = results
                return results
        
    
    #############################
    # Helper and debug methods
    #############################
    
    def generate_line_imgs(self):
        pass

    #############################
    ## Experimental
    #############################
    def viterbi_post_process(self, img_arr, results):
        '''Go through all results and attempts to correct invalid syllables'''
        final = [[] for i in range(len(results))]
        for i, line in enumerate(results):
            syllable = []
            for j, char in enumerate(line):
                if char[-1] in u'་། ' or not word_parts_set.intersection(char[-1]) or j == len(line)-1:
                    if syllable:
                        syl_str = ''.join(s[-1] for s in syllable)
                        
                        if is_non_std(syl_str) and syl_str not in syllables:
                            print syl_str, 'HAS PROBLEMS. TRYING TO FIX'
                            bx = combine_many_boxes([ch[0:4] for ch in syllable])
                            bx = list(bx)

                            arr = img_arr[bx[1]:bx[1]+bx[3], bx[0]:bx[0]+bx[2]]
                            arr = fadd_padding(arr, 3)

                            try:
                                temp_dir = tempfile.mkdtemp()
                                tmpimg = os.path.join(temp_dir, 'tmp.tif')
                                Image.fromarray(arr*255).convert('L').save(tmpimg)
                                pgrec = PageRecognizer(tmpimg, Config(line_break_method='line_cut', page_type='book', postprocess=False, viterbi_postprocessing=True, clear_hr=False, detect_o=False))
                                prob, hmm_res = pgrec.recognize_page()
                                os.remove(tmpimg)
                                os.removedirs(temp_dir)
                            except TypeError:
                                print 'HMM run exited with an error.'
                                prob = 0
                                hmm_res = ''
                            
                            logging.info(u'VPP Correction: %s\t%s' % (syl_str, hmm_res))
                            if prob == 0 and hmm_res == '':
                                print 'hit problem. using unmodified output'
                                for s in syllable:
                                    final[i].append(s)
                            else:
                                bx.append(prob)
                                bx.append(hmm_res)
                                final[i].append(bx)
                        else:
                            for s in syllable:
                                final[i].append(s)
                    final[i].append(char)
                    syllable = []
                else:
                    syllable.append(char)
            if syllable:
                for s in syllable:
                    final[i].append(s)
    
        return final
    
def generate_formatted_page(page_info):
    pass    

def run_recognize(imagepath):
    global args
    command_args = args
    if command_args.conf:
        conf_dict = load_config(command_args.conf)
    else:
        conf_dict = default_config
        
    # Override any confs with command line versions
    for key in conf_dict:

        if not hasattr(command_args, key):
            continue
        val = getattr(command_args, key)
        if val:
            conf_dict[key] = val
            
    rec = PageRecognizer(imagepath, conf=Config(**conf_dict))
    if args.format == 'text':
        text = True
    else:
        text = False
    return rec.recognize_page(text=text)

def run_recognize_remote(imagepath, conf_dict, text=False):
    rec = PageRecognizer(imagepath, conf=Config(**conf_dict))
    results = rec.recognize_page(text=text)
    return results

if __name__ == '__main__':
    DEFAULT_OUTFILE = 'ocr_output.txt'
    
    parser = argparse.ArgumentParser(description='Namsel OCR')
    
    action_choices = ['preprocess', 'recognize-page', 'isolate-lines', 'view-page-info',
                      'recognize-volume']
    parser.add_argument('action', type=str, choices=action_choices,
                        help='The Namsel function to be executed')
    parser.add_argument('imagepath', type=str, help="Path to jpeg, tiff, or png image (or a folder containing them, in the case of recognize-volume)")
    parser.add_argument('--conf', type=str, help='Path to a valid configuration file')
    parser.add_argument('--format', type=str, choices=['text', 'page-info'], help='Format returned by the recogizer')
    parser.add_argument('--outfile', type=str, help='Name of the file saved in the ocr_ouput folder. If not specified, filename will be "ocr_output.txt"')
    # Config override options
    confgroup = parser.add_argument_group('Config', 'Namsel options')
    confgroup.add_argument('--page_type', type=str, choices=['pecha', 'book'], help='Type of page')
    confgroup.add_argument('--line_break_method', type=str, choices=['line_cluster', 'line_cut'],
                           help='Line breaking method. Use line_cluster for page type "pecha"')
    confgroup.add_argument('--recognizer', type=str, choices=['hmm', 'probout'],
                           help='The recognizer to use. Use HMM unless page contains many hard-to-segment and unusual characters')
    confgroup.add_argument('--break_width', type=float, help='Threshold value to determine segmentation, measured in stdev above the mean char width')
    confgroup.add_argument('--segmenter', type=str, help='Type of segmenter to use', choices=['stochastic', 'experimental'])
    confgroup.add_argument('--low_ink', type=bool, help='Attempt to enhance results for poorly inked prints')
    confgroup.add_argument('--line_cluster_pos', type=str, choices=['top', 'center'])
    confgroup.add_argument('--postprocess', type=bool, help='Run viterbi post-processing')
    confgroup.add_argument('--detect_o', type=bool, help='Detect and set aside na-ro vowels in first pass recognition')
    confgroup.add_argument('--clear_hr', type=bool, help='Clear all content above a horizontal rule on top of a page')
    confgroup.add_argument('--line_cut_inflation', type=int, help='The number of iterations to use when dilating image in line breaking. Increase this value when you want to blob things together')
    
    scantailor_conf = parser.add_argument_group('Scantailor', 'Preprocessing options')
    scantailor_conf.add_argument('--layout', choices=['single', 'double'], type=str,
                                 help='Option for telling scantailor to expect double or single pages')
    
    scantailor_conf.add_argument('--threshold', type=int, help="The amount of thinning or thickening of the output of scantailor. Good values are -40 to 40 (for thinning and thickening respectively)")

    args = parser.parse_args()
    
    if not os.path.exists('ocr_results'):
        os.mkdir('ocr_results')
    
    if args.outfile:
        outfilename = args.outfile
    else:
        outfilename = DEFAULT_OUTFILE
    
    if args.action == 'recognize-page':
        results = run_recognize(args.imagepath)
        if args.format == 'text':
            with codecs.open(outfilename, 'w', 'utf-8') as outfile:
                outmessage = '''OCR text\n\n'''
                outfile.write(outmessage)
                outfile.write(os.path.basename(args.imagepath)+'\n')
                if not isinstance(results, (str, unicode)):
                    results = 'No content captured for this image'
                    print '****************'
                    print results
                    print "Saving empty page to output"
                    print '****************'
                outfile.write(results)

    elif args.action == 'recognize-volume':
        import multiprocessing
        import glob
        if not os.path.isdir(args.imagepath):
            print 'Error: You must specify the name of a directory containing tif images in order to recognize a volume'
            sys.exit()
        pool = multiprocessing.Pool()
        pages = glob.glob(os.path.join(args.imagepath, '*tif'))
        pages.sort()
        results = pool.map(run_recognize,  pages)
        if args.format == 'text':

            with codecs.open(outfilename, 'w', 'utf-8') as outfile:
                outmessage = '''OCR text\n\n'''
                outfile.write(outmessage)

                for k, r in enumerate(results):
                    outfile.write(os.path.basename(pages[k])+'\n')
                    if not isinstance(r, str) and not isinstance(r, unicode):
                        r = 'No content captured'

                    outfile.write(r + '\n\n')
    elif args.action == 'preprocess':
        run_scantailor(args.imagepath, args.threshold, layout=args.layout)
