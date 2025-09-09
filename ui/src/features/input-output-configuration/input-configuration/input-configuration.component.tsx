/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { GenICam, ImagesFolder, IPCamera, VideoFile, WebCam } from '@geti-prompt/icons';

import { DisclosureGroup } from '../ui/disclosure-group/disclosure-group.component';

const inputs = [
    { label: 'Webcam', value: 'webcam', content: 'Test', icon: <WebCam width={'24px'} /> },
    { label: 'IP Camera', value: 'ip-camera', content: 'Test', icon: <IPCamera width={'24px'} /> },
    { label: 'GenICam', value: 'gen-i-cam', content: 'Test', icon: <GenICam width={'24px'} /> },
    { label: 'Video file', value: 'video-file', content: 'Test', icon: <VideoFile width={'24px'} /> },
    { label: 'Image folder', value: 'image-folder', content: 'Test', icon: <ImagesFolder width={'24px'} /> },
];

export const InputConfiguration = () => {
    const [selectedInput, setSelectedInput] = useState<string | null>(null);

    return <DisclosureGroup onChange={setSelectedInput} items={inputs} value={selectedInput} />;
};
