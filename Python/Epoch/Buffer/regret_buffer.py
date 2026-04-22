"""    
Regret Buffer (RGSC)
        Maintains a separate, prioritized subset of high-error (regret) states.
        Used by SelfPlay to seed PRB restarts (forcing the model to practice tricky mid-game positions instead of always starting from an empty board).
      
        
ANY REGRET BUFFER IMPLEMENTATION MUST EXACTLY MATCH THE SPECS IN
/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/Docs/2602.20809v1.txt
              """

