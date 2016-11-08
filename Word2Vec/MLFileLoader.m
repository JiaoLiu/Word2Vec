//
//  MLFileLoader.m
//  Word2Vec
//
//  Created by Jiao Liu on 11/2/16.
//  Copyright © 2016 ChangHong. All rights reserved.
//

#import "MLFileLoader.h"

@implementation MLFileLoader

- (id)initWithFile:(NSString *)filePath
{
    self = [super init];
    if (self) {
        _filePath = filePath;
        _indexDict = [NSMutableDictionary dictionary];
        _dictionary = calloc(max_dicSize, sizeof(struct dic_word));
    }
    return self;
}

- (id)init
{
    self = [super init];
    if (self) {
        _indexDict = [NSMutableDictionary dictionary];
        _dictionary = calloc(max_dicSize, sizeof(struct dic_word));
    }
    return self;
}

#pragma mark - Interface


- (int)readWordIndex:(FILE *)file
{
    char word[MAX_WORDLENGTH];
    [self readWord:word File:file];
    if (feof(file)) {
        return -1;
    }
    return [self findWordInDict:word];
}

#pragma mark - Load Methods

- (void)readWordFromFile
{
    FILE *file = fopen([_filePath UTF8String], "rb");
    char word[MAX_WORDLENGTH];
    _dicSize = 0;
    _totalWords = 0;
    if (file == NULL) {
        NSLog(@"File Error");
        return;
    }
    while (1) {
        [self readWord:word File:file];
        if (feof(file)) {
            break;
        }
        if (strlen(word) == 0) {
            continue;
        }
        int i = [self findWordInDict:word];
        if (i == -1) { // not found
            [self addWordToDict:word];
        }
        else
        {
            _dictionary[i].frequency++;
        }
        _totalWords++;
    }
    [self sortDict];
    [self buildHuffmanTree];
    _fileSize = ftell(file);
    fclose(file);
}

- (void)readWord:(char *)word File:(FILE *)file
{
    int letter;
    int len = 0;
    while (!feof(file)) {
        letter = fgetc(file);
        if (letter == 13) { // New Page
            continue;
        }
        if (len == 0 && letter == '\n') { // ignore Return
            word[len] = 0;
            return;
        }
        if (letter == ' ' || letter == '\n' || letter == '\t') { // ignore Return + Space + EOL
            break;
        }
        if (len >= MAX_WORDLENGTH - 1) {
            break;
        }
        word[len] = letter;
        len++;
    }
    word[len] = 0;
}

- (int)findWordInDict:(char *)word
{
    id index = [_indexDict objectForKey:[[NSString stringWithCString:word encoding:NSUTF8StringEncoding] lowercaseString]];
    if (index == NULL) {
        return -1;
    }
    return [index intValue];
}

- (void)addWordToDict:(char *)word
{
    long len = strlen(word) + 1;
    _dictionary[_dicSize].word = calloc(len, (sizeof(char)));
    strcpy(_dictionary[_dicSize].word, word);
    _dictionary[_dicSize].frequency = 1;
    _dicSize++;
    
    if (_dicSize +  2 > max_dicSize) {
        max_dicSize += 1000;
        _dictionary = realloc(_dictionary, sizeof(struct dic_word) *  max_dicSize);
    }
    
    [_indexDict setValue:[NSNumber numberWithInt:_dicSize-1] forKey:[[NSString stringWithCString:word encoding:NSUTF8StringEncoding] lowercaseString]];
}

int cmp(const void *a, const void *b)
{
    return (int)(((struct dic_word *)b)->frequency - ((struct dic_word *)a)->frequency);
}

- (void)sortDict
{
    qsort(&_dictionary[0], _dicSize, sizeof(struct dic_word), cmp);
    [_indexDict removeAllObjects];
    for (int i = 0; i < _dicSize; i++) {
        [_indexDict setValue:[NSNumber numberWithInt:i] forKey:[[NSString stringWithCString:_dictionary[i].word encoding:NSUTF8StringEncoding] lowercaseString]];
        _dictionary[i].code = calloc(MAX_CODELENGTH, sizeof(int));
        _dictionary[i].position = calloc(MAX_CODELENGTH, sizeof(int));
    }
}

- (void)buildHuffmanTree
{
    int size = _dicSize * 2;
    long long code[MAX_CODELENGTH], position[MAX_CODELENGTH];
    long long *freq = calloc(size + 1, sizeof(long long));
    long long *parent = calloc(size + 1, sizeof(long long));
    long long *weight = calloc(size + 1, sizeof(long long));
    for (int i = 0; i < _dicSize; i++) {
        freq[i] = _dictionary[i].frequency;
    }
    for (int i = _dicSize; i < size; i++) {
        freq[i] = 1e15;
    }
    
    long long pos1 = _dicSize - 1;
    long long pos2 = _dicSize;
    long long min1, min2;
    for (int i = 0; i < _dicSize - 1; i++) {
        if (pos1 >= 0) {  // find left node
            if (freq[pos1] < freq[pos2]) {
                min1 = pos1;
                pos1--;
            }
            else
            {
                min1 = pos2;
                pos2++;
            }
        }
        else
        {
            min1 = pos2;
            pos2++;
        }

        if (pos1 >= 0) { // find right node
            if (freq[pos1] < freq[pos2]) {
                min2 = pos1;
                pos1--;
            }
            else
            {
                min2 = pos2;
                pos2++;
            }
        }
        else
        {
            min2 = pos2;
            pos2++;
        }

        // combine tree
        freq[_dicSize + i] = freq[min1] + freq[min2];
        parent[min1] = parent[min2] = _dicSize + i;
        weight[min1] = 0;
        weight[min2] = 1;
    }
    
    // assign dictionary code
    for (int i = 0; i < _dicSize; i++) {
        long long back = i;
        int pos = 0;
        while (back != size - 2) {
            code[pos] = weight[back];
            position[pos] = back;
            back = parent[back];
            pos++;
        }
        _dictionary[i].codeLength = pos;
        _dictionary[i].position[0] = _dicSize - 2; // position[0] -> the root
        
        // loop: code[0] <-> position[1] ----> code[n] <-> position[n+1]
        for (int j = 0; j < pos; j++) {
            _dictionary[i].code[pos-1-j] = (int)code[j];
            _dictionary[i].position[pos-j] = (int)position[j] - _dicSize;
        }
    }
    
    free(freq);
    free(parent);
    free(weight);
}

@end
