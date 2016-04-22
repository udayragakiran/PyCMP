#include "extcode.h"
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * DAQIOSetAttenuatorLevel
 */
int32_t __cdecl DAQIOSetAttenuatorLevel(uint8_t AttenLevelDB, 
	char AttenuatorLines[]);

MgErr __cdecl LVDLLStatus(char *errStr, int errStrLen, void *module);

#ifdef __cplusplus
} // extern "C"
#endif

