//
//  MLFileLoader.h
//  Word2Vec
//
//  Created by Jiao Liu on 11/2/16.
//  Copyright © 2016 ChangHong. All rights reserved.
//

#import <Foundation/Foundation.h>

#define MAX_WORDLENGTH 100
#define MAX_CODELENGTH 40

struct dic_word {
    int *code; // Huffman Code
    int codeLength;
    int *position; // path of the HuffmanCode
    char *word;
    long long frequency;
};

//const int hashSize = 3 * 1e8;
static int max_dicSize = 1000;

@interface MLFileLoader : NSObject
{
    NSMutableDictionary *_indexDict;
}

- (id)initWithFile:(NSString *)filePath;
- (void)readWordFromFile;
- (int)readWordIndex:(FILE *)file;
- (int)findWordInDict:(char *)word;

@property (nonatomic, strong)NSString *filePath;
@property (nonatomic, readonly)long long totalWords;
@property (nonatomic, readonly)long long fileSize;
@property (nonatomic, readonly)int dicSize;
@property (nonatomic, readonly)struct dic_word *dictionary;

@end
