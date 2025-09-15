/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { Button, Flex, Form, Item, Picker, TextField } from '@geti/ui';

export const IPCameraForm = () => {
    const [ip, setIp] = useState('');
    const [port, setPort] = useState('');
    const [streamPath, setStreamPath] = useState('');

    const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
    };

    const protocols = ['RTSP', 'HTTP', 'HTTPS', 'ONVIF', 'SMTP', 'TCP'];

    const isFormValid = !ip || !port || !streamPath;

    return (
        <Form onSubmit={handleSubmit}>
            <Flex direction='row' gap='size-200'>
                <TextField label='IP Address' value={ip} onChange={setIp} />
                <TextField label='Port' value={port} onChange={setPort} />
            </Flex>
            <TextField label='Stream Path' value={streamPath} onChange={setStreamPath} />
            <Picker label='Protocol' defaultSelectedKey={protocols[0]} items={protocols}>
                {protocols.map((protocol) => (
                    <Item key={protocol}>{protocol}</Item>
                ))}
            </Picker>
            <Button type='submit' width={'size-700'} isDisabled={isFormValid}>
                Apply
            </Button>
        </Form>
    );
};
