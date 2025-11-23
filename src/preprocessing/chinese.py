"""
Chinese text preprocessing module.

Uses Jieba for tokenization with open-source Chinese stopwords.
"""

import re
import jieba
import jieba.analyse
from typing import List, Set, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Chinese stopwords list (open-source)
CHINESE_STOPWORDS = {
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你',
    '会', '着', '没有', '看', '好', '自己', '这', '那', '他', '她', '它', '们', '个', '中', '为', '大', '来', '以', '到', '及',
    '也', '都', '就', '而', '或', '但', '与', '等', '能', '对', '可', '可以', '会', '可能', '应该', '必须', '需要', '要', '想',
    '如果', '因为', '所以', '虽然', '但是', '然而', '不过', '而且', '并且', '或者', '还是', '还是', '还是', '还是', '还是',
    '这个', '那个', '这些', '那些', '这样', '那样', '这里', '那里', '这时', '那时', '现在', '以前', '以后', '今天', '明天', '昨天',
    '年', '月', '日', '时', '分', '秒', '点', '刻', '钟', '小时', '分钟', '秒钟', '天', '周', '星期', '年', '月', '日',
    '什么', '怎么', '为什么', '哪里', '哪个', '哪些', '多少', '几', '多', '少', '大', '小', '长', '短', '高', '低', '好', '坏',
    '新', '旧', '老', '年轻', '快', '慢', '早', '晚', '多', '少', '大', '小', '长', '短', '高', '低', '好', '坏', '新', '旧',
    '我', '你', '他', '她', '它', '我们', '你们', '他们', '她们', '它们', '自己', '别人', '大家', '人家', '别人', '自己',
    '是', '不是', '有', '没有', '会', '不会', '能', '不能', '可以', '不可以', '应该', '不应该', '必须', '不必', '需要', '不需要',
    '做', '作', '用', '使', '让', '给', '把', '被', '由', '从', '向', '往', '到', '在', '于', '对', '对于', '关于', '至于',
    '和', '与', '及', '以及', '或', '或者', '还是', '但是', '不过', '然而', '而且', '并且', '所以', '因此', '因为', '由于',
    '如果', '假如', '要是', '只要', '只有', '除非', '无论', '不管', '尽管', '虽然', '即使', '就算', '哪怕', '纵然',
    '的', '地', '得', '了', '着', '过', '呢', '吗', '吧', '啊', '呀', '啦', '嘛', '哦', '嗯', '唉', '哎', '喂', '嘿', '嗨',
}


class ChinesePreprocessor:
    """Preprocessor for Chinese text using Jieba."""
    
    def __init__(self, remove_stopwords: bool = True, use_custom_stopwords: bool = True):
        """
        Initialize Chinese preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            use_custom_stopwords: Whether to use custom stopwords list
        """
        self.remove_stopwords = remove_stopwords
        self.stop_words: Set[str] = set()
        
        if remove_stopwords:
            if use_custom_stopwords:
                self.stop_words = CHINESE_STOPWORDS.copy()
            else:
                # Try to load from file if available
                self._load_stopwords()
    
    def _load_stopwords(self, stopwords_file: Optional[Path] = None):
        """Load stopwords from file if available."""
        # This can be extended to load from a file
        pass
    
    def normalize(self, text: str) -> str:
        """
        Normalize text by removing special characters and extra whitespace.
        
        Args:
            text: Input text string
            
        Returns:
            Normalized text string
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', '', text)  # Chinese text typically has no spaces
        
        # Remove special characters (keep Chinese characters, numbers, basic punctuation)
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：""''（）【】]', '', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Chinese text using Jieba.
        
        Args:
            text: Input text string
            
        Returns:
            List of token strings
        """
        # Use Jieba for word segmentation
        tokens = jieba.cut(text, cut_all=False)
        tokens = list(tokens)
        
        return tokens
    
    def remove_stopwords_func(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of token strings
            
        Returns:
            List of tokens with stopwords removed
        """
        if not self.remove_stopwords:
            return tokens
        
        return [token for token in tokens if token not in self.stop_words and len(token.strip()) > 0]
    
    def preprocess(self, text: str) -> List[str]:
        """
        Complete preprocessing pipeline: normalize, tokenize, remove stopwords.
        
        Args:
            text: Input text string
            
        Returns:
            List of preprocessed tokens
        """
        # Normalize
        text = self.normalize(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        tokens = self.remove_stopwords_func(tokens)
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def preprocess_to_string(self, text: str) -> str:
        """
        Preprocess text and return as space-separated string.
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text string (space-separated tokens)
        """
        tokens = self.preprocess(text)
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input text strings
            
        Returns:
            List of preprocessed token lists
        """
        return [self.preprocess(text) for text in texts]
    
    def preprocess_batch_to_strings(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts and return as strings.
        
        Args:
            texts: List of input text strings
            
        Returns:
            List of preprocessed text strings (space-separated tokens)
        """
        return [self.preprocess_to_string(text) for text in texts]


def preprocess_chinese_text(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Convenience function to preprocess a single Chinese text.
    
    Args:
        text: Input text string
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        List of preprocessed tokens
    """
    preprocessor = ChinesePreprocessor(remove_stopwords=remove_stopwords)
    return preprocessor.preprocess(text)


def preprocess_chinese_batch(texts: List[str], remove_stopwords: bool = True) -> List[List[str]]:
    """
    Convenience function to preprocess a batch of Chinese texts.
    
    Args:
        texts: List of input text strings
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        List of preprocessed token lists
    """
    preprocessor = ChinesePreprocessor(remove_stopwords=remove_stopwords)
    return preprocessor.preprocess_batch(texts)
