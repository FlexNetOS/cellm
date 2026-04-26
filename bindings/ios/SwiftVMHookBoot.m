//  SwiftVMHookBoot.m
//  Companion to SwiftVMHook.swift — ensures the hook auto-starts.
//  __attribute__((constructor)) runs at module load, before main().
//  This is needed because Swift global variables are lazily initialized.

#if DEBUG
#import <dispatch/dispatch.h>

extern void _swiftvm_hook_init(void);

__attribute__((constructor))
static void _swiftvm_hook_boot(void) {
    dispatch_async(dispatch_get_main_queue(), ^{
        _swiftvm_hook_init();
    });
}
#endif
